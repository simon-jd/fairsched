"""Run one (workload, arm, seed) experiment end to end on the local GPU.

    python3 -m bench.run noisy_neighbor vtc --seed 1
    python3 -m bench.run aggressive native fcfs vtc --name overhead --seed 1

Starts the pinned vLLM container (with the plugin, or native for ``native``),
the Go gateway in front of it and the Go load generator, then tears everything
down. Raw output goes to ``results/raw/<name>/<arm>-s<seed>/``, where the name
defaults to the workload.

Arms:
  native      vLLM FCFS, no plugin (overhead reference)
  fcfs        vLLM FCFS with the plugin in passthrough mode (same accounting/logging)
  fcfs-cap    fcfs, and the gateway admits at most CAP_INFLIGHT requests per
              tenant to the engine; the rest wait in the gateway's FIFO queue
  priority    vLLM priority scheduling, gateway gives tenant A the higher priority
  priority-b  the same with tenant B favoured
  vtc         the plugin's VTC policy
  vtc-kv      VTC with preempt_order "kv_share" (vLLM's KV-pressure victim comes
              from a tenant holding more than its share of the running KV)
  vtc-fit     VTC with the decode weight from the step-time fit (FIT_W_D)
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from bench.workloads import MODEL, WORKLOADS, write as write_trace

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "configs/pins.json").read_text())
# Run the image by digest, so a re-pushed tag cannot change what is measured.
IMAGE = PINS["image"].split(":")[0] + "@" + PINS["image_digest"]
ARMS = ("native", "fcfs", "fcfs-cap", "priority", "priority-b", "vtc", "vtc-kv", "vtc-fit")
# Decode weight of the vtc-fit arm, prefill weight 1: the fitted marginal cost of
# a decode step over that of a prefill token on mixed_shapes (0.0789 / 0.0809 ms,
# results/mixed_shapes/gpu_time.json). VTC's default is 2.
FIT_W_D = 0.97
# Per-tenant in-flight cap of the fcfs-cap arm: 16 of noisy_neighbor's B requests
# (2,304 tokens each) hold about 37k tokens, under half of the 84k-token KV cache.
CAP_INFLIGHT = 16
ENGINE_PORT, PROXY_PORT = 8000, 8080
CONTAINER = "fairsched-engine"
# Other projects on this machine queue for the GPU on the same lock file.
GPU_LOCK = Path(os.environ.get("FAIRSCHED_GPU_LOCK", "/var/tmp/rtx4090.lock"))
# The directories whose code a run executes; their git tree hashes identify it.
SOURCE_DIRS = ("fairsched", "bench", "cmd", "proxy", "loadgen", "configs")


@contextmanager
def gpu_lock():
    """Block until this process holds the machine-wide GPU lock."""
    with os.fdopen(os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o666), "a+") as lock:
        print(f"waiting for {GPU_LOCK} ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        print("GPU lock acquired", flush=True)
        yield


def gpu_is_free(limit_mib: int = 1024) -> None:
    """Refuse to start if another workload holds GPU memory (desktop apps use ~0.3 GiB)."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    busy = [line for line in out.splitlines() if line.strip() and int(line.rsplit(",", 1)[1]) > limit_mib]
    if busy:
        sys.exit(f"GPU is in use, not starting: {busy}")


def source_tree() -> tuple[dict[str, str], bool]:
    """Tree hash of each source directory at HEAD, and whether any has uncommitted changes."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    trees = {d: git("rev-parse", f"HEAD:{d}") for d in SOURCE_DIRS}
    return trees, bool(git("status", "--porcelain", "--", *SOURCE_DIRS))


def engine_command(arm: str, workload: str, run: Path) -> list[str]:
    hf = ROOT / ".cache/huggingface"
    vllm_cache = ROOT / ".cache/vllm"
    vllm_cache.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "-d", "--name", CONTAINER, "--gpus", "all", "--ipc=host",
           "-p", f"127.0.0.1:{ENGINE_PORT}:8000",
           "-v", f"{ROOT}:/workspace:ro", "-v", f"{run}:/fairsched-output", "-v", f"{hf}:/root/.cache/huggingface:ro",
           "-v", f"{vllm_cache}:/root/.cache/vllm",
           "-e", "HF_HUB_OFFLINE=1", "-e", "PYTHONPATH=/workspace", "-e", "PYTHONDONTWRITEBYTECODE=1",
           "-e", "FAIRSCHED_OUTPUT=/fairsched-output", "-e", "FAIRSCHED_POLICY=/fairsched-output/policy.json",
           "-e", f"FAIRSCHED_MODE={'vtc' if arm.startswith('vtc') else 'passthrough'}",
           IMAGE, "--model", MODEL, "--revision", PINS["models"][MODEL],
           "--dtype", "bfloat16", "--max-model-len", "8192",
           # Desktop apps hold ~3.5 GiB of this 24 GiB card.
           "--gpu-memory-utilization", "0.83",
           "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--enable-prefix-caching",
           "--generation-config", "vllm",
           "--host", "0.0.0.0", "--port", "8000"]
    if arm != "native":
        cmd += ["--scheduler-cls", "fairsched.TenantAsyncScheduler"]
    if arm.startswith("priority"):
        cmd += ["--scheduling-policy", "priority"]
    if workload == "long_prompt":
        # About 8,200 tokens of KV, the size of the run that livelocked the old
        # policy: one 6,272-token request nearly fills it, so running requests
        # contend for blocks and vLLM preempts. vLLM keeps one block back (the
        # null block), so 512 would not fit one max-length (8,192) request.
        cmd += ["--num-gpu-blocks-override", "520"]
    return cmd


def policy_config(arm: str, workload: str) -> dict:
    """The plugin's policy file for one run: the workload's weights, plus the arm's options."""
    config = {"tenants": {t: {"weight": w} for t, w in WORKLOADS[workload].weights.items()}}
    if arm == "vtc-kv":
        config["preempt_order"] = "kv_share"
    if arm == "vtc-fit":
        config["w_d"] = FIT_W_D
    return config


def proxy_config(arm: str, run: Path) -> Path:
    tenant = {"default_deadline_ms": 0, "max_inflight": CAP_INFLIGHT if arm == "fcfs-cap" else 1024,
              "max_queued": 1024}
    config = {"listen": f"127.0.0.1:{PROXY_PORT}", "write_timeout_ms": 900_000,
              "upstream": f"http://127.0.0.1:{ENGINE_PORT}",
              "api_keys": {"key-A": "A", "key-B": "B"},
              "tenants": {"A": {**tenant, "class": "interactive", "priority": 0 if arm == "priority-b" else -1},
                          "B": {**tenant, "class": "batch", "priority": -1 if arm == "priority-b" else 0}},
              "log_path": str(run / "proxy.jsonl"), "priority_baseline": arm.startswith("priority")}
    path = run / "proxy.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def wait_healthy(timeout_s: float = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
                               capture_output=True, text=True).stdout.strip()
        if state != "true":
            raise RuntimeError("engine container exited during startup; see engine.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{ENGINE_PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(2)
    raise TimeoutError("engine did not become healthy")


def warm_up() -> None:
    """Two tiny requests through the gateway, so CUDA graphs and the HTTP path are hot."""
    for key in ("key-A", "key-B"):
        body = json.dumps({"model": MODEL, "prompt": [1000 + i for i in range(32)], "max_tokens": 8,
                           "ignore_eos": True}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{PROXY_PORT}/v1/completions", data=body,
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()


def run_one(workload: str, arm: str, seed: int, name: str | None = None) -> Path:
    if arm not in ARMS or workload not in WORKLOADS:
        raise SystemExit(f"unknown arm/workload: {arm}/{workload}")
    run = ROOT / "results/raw" / (name or workload) / f"{arm}-s{seed}"
    if run.exists():
        raise SystemExit(f"{run} exists; remove it to rerun")
    run.mkdir(parents=True)
    os.chmod(run, 0o777)       # the engine container writes here as root
    with gpu_lock():
        gpu_is_free()
        return _run_locked(workload, arm, seed, run)


def _run_locked(workload: str, arm: str, seed: int, run: Path) -> Path:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    trace = write_trace(workload, seed, run / "trace.jsonl")
    (run / "policy.json").write_text(json.dumps(policy_config(arm, workload)))
    trees, dirty = source_tree()
    meta = {"workload": workload, "arm": arm, "seed": seed, "model": MODEL,
            "source_tree": trees, "source_dirty": dirty,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(),
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "engine_command": engine_command(arm, workload, run)}
    subprocess.run(meta["engine_command"], check=True, capture_output=True)
    gpu_log = open(run / "gpu.csv", "w")
    proxy = loadgen = gpu = None
    try:
        wait_healthy()
        proxy = subprocess.Popen([str(ROOT / "bin/fairsched-proxy"), "-config", str(proxy_config(arm, run))],
                                 stdout=open(run / "proxy.out", "w"), stderr=subprocess.STDOUT)
        time.sleep(1)
        warm_up()
        gpu = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
                                "--format=csv,noheader,nounits", "-lms", "500"], stdout=gpu_log)
        meta["trace_start_unix"] = time.time()
        loadgen = subprocess.run([str(ROOT / "bin/fairsched-loadgen"), "-url", f"http://127.0.0.1:{PROXY_PORT}",
                                  "-trace", str(trace), "-output", str(run / "client.jsonl"),
                                  "-report", str(run / "client-report.json"), "-timeout", "15m"],
                                 capture_output=True, text=True)
        meta["trace_end_unix"] = time.time()
        meta["loadgen_returncode"] = loadgen.returncode
        (run / "loadgen.out").write_text(loadgen.stdout + loadgen.stderr)
    finally:
        for p in (gpu, proxy):
            if p is not None:
                p.send_signal(signal.SIGTERM)
                p.wait(timeout=30)
        subprocess.run(["docker", "stop", "-t", "60", CONTAINER], capture_output=True)
        with open(run / "engine.log", "w") as log:
            subprocess.run(["docker", "logs", CONTAINER], stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workload", choices=sorted(WORKLOADS))
    parser.add_argument("arms", nargs="+", choices=ARMS)
    parser.add_argument("--seed", type=int, action="append", help="repeatable; default 1")
    parser.add_argument("--name", help="directory under results/raw (default: the workload)")
    args = parser.parse_args()
    for seed in args.seed or [1]:
        for arm in args.arms:
            print(f"== {args.workload} {arm} seed={seed}", flush=True)
            print(run_one(args.workload, arm, seed, args.name), flush=True)


if __name__ == "__main__":
    main()
