"""Run one (workload, arm, seed) experiment on several vLLM replicas behind the gateway.

    python3 -m bench.replicas aggressive gvtc-sticky --seed 1 --gpus 0,1 \\
        --model /models/qwen2.5-7b --served-model Qwen/Qwen2.5-7B-Instruct --out results/raw/replicas

Unlike ``bench.run`` (one engine in the pinned container), this starts each
engine as a ``vllm serve`` process from the given environment, one per entry
of ``--gpus`` (repeat an index to put two small engines on one GPU), the
counter service when the arm shares counters, the gateway with one upstream
per engine, and the load generator; then it shuts every process down.

Arms, each a scheduler policy and a routing:
  fcfs-rr, fcfs-sticky    vLLM FCFS with the plugin in passthrough mode
  vtc-rr, vtc-sticky      VTC with each engine's own counters (the default)
  gvtc-rr, gvtc-sticky    VTC with counters shared through fairsched.counters
``rr`` alternates each tenant's requests over the engines; ``sticky`` hashes
each request's session, and the trace gives all of A's sessions to engine 0
and spreads B's over all engines.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from bench.workloads import WORKLOADS, write as write_trace

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("fcfs", "vtc", "gvtc")
ROUTINGS = {"rr": "round_robin", "sticky": "sticky"}
ARMS = tuple(f"{p}-{r}" for p in POLICIES for r in ROUTINGS)
GPU_QUERY = "index,timestamp,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate"
# Under sticky routing all of A's sessions hash to the first engine.
PINNED = {"A": 0}


@contextmanager
def file_lock(path: str):
    with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o666), "a+") as lock:
        print(f"waiting for {path} ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def run_name(arm: str, seed: int, variant: str | None) -> str:
    return f"{arm}{'-' + variant if variant else ''}-s{seed}"


def engine_env(args, arm: str, run: Path, index: int, gpu: str) -> dict[str, str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               FAIRSCHED_OUTPUT=str(run / f"replica{index}"), FAIRSCHED_POLICY=str(run / "policy.json"),
               FAIRSCHED_MODE="passthrough" if arm.startswith("fcfs") else "vtc",
               FAIRSCHED_REPLICA=f"r{index}")
    for name in ("FAIRSCHED_COUNTER_URL", "FAIRSCHED_SYNC_MS"):
        env.pop(name, None)
    if arm.startswith("gvtc"):
        env["FAIRSCHED_COUNTER_URL"] = f"http://127.0.0.1:{args.counter_port}"
        env["FAIRSCHED_SYNC_MS"] = str(args.sync_ms)
    return env


def engine_command(args, index: int) -> list[str]:
    return [args.vllm, "serve", args.model, "--served-model-name", args.served_model,
            "--dtype", "bfloat16", "--max-model-len", "8192",
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--enable-prefix-caching",
            "--generation-config", "vllm", "--scheduler-cls", "fairsched.TenantAsyncScheduler",
            "--host", "127.0.0.1", "--port", str(args.engine_port + index), *args.engine_arg]


def proxy_config(args, arm: str, run: Path, engines: int) -> Path:
    tenant = {"default_deadline_ms": 0, "max_inflight": 1024, "max_queued": 1024}
    config = {"listen": f"127.0.0.1:{args.proxy_port}", "write_timeout_ms": 900_000,
              "upstreams": [f"http://127.0.0.1:{args.engine_port + i}" for i in range(engines)],
              "routing": ROUTINGS[arm.split("-", 1)[1]],
              "api_keys": {"key-A": "A", "key-B": "B"},
              "tenants": {"A": {**tenant, "class": "interactive"}, "B": {**tenant, "class": "batch"}},
              "log_path": str(run / "proxy.jsonl")}
    path = run / "proxy.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def get(url: str, timeout: float = 2) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status


def wait_healthy(engines: list[subprocess.Popen], ports: list[int], timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    pending = dict(zip(ports, engines))
    while pending and time.time() < deadline:
        for port, proc in list(pending.items()):
            if proc.poll() is not None:
                raise RuntimeError(f"engine on port {port} exited with {proc.returncode}; see its log")
            try:
                if get(f"http://127.0.0.1:{port}/health") == 200:
                    del pending[port]
            except OSError:
                pass
        time.sleep(2)
    if pending:
        raise TimeoutError(f"engines on ports {sorted(pending)} did not become healthy")


def warm_up(port: int, served_model: str) -> None:
    """One short request straight to an engine (no tenant, billed to "default")."""
    body = json.dumps({"model": served_model, "prompt": [1000 + i for i in range(32)], "max_tokens": 8,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()


def stop(proc: subprocess.Popen | None, grace_s: float = 60) -> None:
    """SIGTERM the process group, wait, then SIGKILL whatever is left (EngineCore included)."""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def source_commit() -> str:
    commit = os.environ.get("FAIRSCHED_SOURCE_COMMIT")
    if commit:
        return commit
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "unknown"


def run_one(args, workload: str, arm: str, seed: int) -> Path:
    if arm not in ARMS or workload not in WORKLOADS:
        raise SystemExit(f"unknown arm/workload: {arm}/{workload}")
    run = Path(args.out) / run_name(arm, seed, args.variant)
    if run.exists():
        raise SystemExit(f"{run} exists; remove it to rerun")
    run.mkdir(parents=True)
    with file_lock(args.gpu_lock) if args.gpu_lock else nullcontext():
        return _run(args, workload, arm, seed, run)


def _run(args, workload: str, arm: str, seed: int, run: Path) -> Path:
    gpus = args.gpus.split(",")
    trace = write_trace(workload, seed, run / "trace.jsonl", model=args.served_model, replicas=len(gpus),
                        pinned=PINNED)
    (run / "policy.json").write_text(json.dumps(
        {"tenants": {t: {"weight": w} for t, w in WORKLOADS[workload].weights.items()}}))
    meta = {"workload": workload, "arm": arm, "seed": seed, "variant": args.variant, "replicas": len(gpus),
            "gpus": gpus, "model": args.model, "served_model": args.served_model, "git_commit": source_commit(),
            "sync_ms": args.sync_ms if arm.startswith("gvtc") else None,
            "kill_counters_at_s": args.kill_counters_at, "pinned_sessions": PINNED,
            "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "engine_command": engine_command(args, 0),
            "nvidia_smi": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()}
    procs: dict[str, subprocess.Popen | None] = {"counters": None, "proxy": None, "gpu": None}
    engines: list[subprocess.Popen] = []
    logs = []

    def spawn(name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
        log = open(run / f"{name}.log", "w")
        logs.append(log)
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                start_new_session=True)

    kill_timer = None
    try:
        if arm.startswith("gvtc"):
            procs["counters"] = spawn("counters", [sys.executable, "-m", "fairsched.counters",
                                                   "--listen", f"127.0.0.1:{args.counter_port}"],
                                      dict(os.environ, PYTHONPATH=str(ROOT)))
        ports = [args.engine_port + i for i in range(len(gpus))]
        # Engines sharing a GPU start one after another, so each profiles its memory alone.
        sequential = len(set(gpus)) < len(gpus)
        started = time.time()
        for i, gpu in enumerate(gpus):
            engines.append(spawn(f"engine{i}", engine_command(args, i), engine_env(args, arm, run, i, gpu)))
            if sequential:
                wait_healthy(engines[-1:], ports[i:i + 1], args.startup_timeout)
        wait_healthy(engines, ports, args.startup_timeout)
        meta["engine_startup_s"] = round(time.time() - started, 1)
        procs["proxy"] = spawn("proxy", [args.proxy, "-config", str(proxy_config(args, arm, run, len(gpus)))])
        time.sleep(1)
        for port in ports:
            warm_up(port, args.served_model)
        procs["gpu"] = subprocess.Popen(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits",
                                         "-lms", "500"], stdout=open(run / "gpu.csv", "w"), start_new_session=True)
        if args.kill_counters_at is not None and procs["counters"] is not None:
            def kill():
                meta["counters_killed_unix"] = time.time()
                os.killpg(procs["counters"].pid, signal.SIGKILL)
            kill_timer = threading.Timer(args.kill_counters_at, kill)
        meta["trace_start_unix"] = time.time()
        if kill_timer is not None:
            kill_timer.start()
        loadgen = subprocess.run([args.loadgen, "-url", f"http://127.0.0.1:{args.proxy_port}", "-trace", str(trace),
                                  "-output", str(run / "client.jsonl"), "-report", str(run / "client-report.json"),
                                  "-timeout", "15m"], capture_output=True, text=True)
        meta["trace_end_unix"] = time.time()
        meta["loadgen_returncode"] = loadgen.returncode
        (run / "loadgen.out").write_text(loadgen.stdout + loadgen.stderr)
    finally:
        if kill_timer is not None:
            kill_timer.cancel()
        for name in ("gpu", "proxy"):
            stop(procs[name], grace_s=30)
        for engine in engines:
            try:
                os.killpg(engine.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for engine in engines:
            stop(engine)
        stop(procs["counters"], grace_s=10)
        for log in logs:
            log.close()
        (run / "meta.json").write_text(json.dumps(meta, indent=2))
    return run


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workload", choices=sorted(WORKLOADS))
    p.add_argument("arms", nargs="+", choices=ARMS)
    p.add_argument("--seed", type=int, action="append", help="repeatable; default 1")
    p.add_argument("--out", default=str(ROOT / "results/raw/replicas"))
    p.add_argument("--variant", help="suffix for the run directory (e.g. t10, kill)")
    p.add_argument("--gpus", default="0,1", help="CUDA device per engine, comma-separated")
    p.add_argument("--model", required=True, help="model directory (or Hugging Face name)")
    p.add_argument("--served-model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--engine-arg", action="append", default=[], help="extra vllm serve argument (repeatable)")
    p.add_argument("--sync-ms", type=float, default=100.0)
    p.add_argument("--kill-counters-at", type=float, help="SIGKILL the counter service this many seconds in")
    p.add_argument("--vllm", default="vllm", help="the vllm executable")
    p.add_argument("--proxy", default=str(ROOT / "bin/fairsched-proxy"))
    p.add_argument("--loadgen", default=str(ROOT / "bin/fairsched-loadgen"))
    p.add_argument("--gpu-lock", help="flock this file for the whole run (the RTX 4090 host's lock)")
    p.add_argument("--startup-timeout", type=float, default=900)
    p.add_argument("--proxy-port", type=int, default=8080)
    p.add_argument("--counter-port", type=int, default=8090)
    p.add_argument("--engine-port", type=int, default=8001, help="first engine's port; the others follow")
    return p


def main() -> None:
    args = parser().parse_args()
    for seed in args.seed or [1]:
        for arm in args.arms:
            print(f"== {args.workload} {arm} seed={seed}", flush=True)
            print(run_one(args, args.workload, arm, seed), flush=True)


if __name__ == "__main__":
    main()
