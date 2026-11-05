"""fairsched across two vLLM 0.31.0 replicas on PORTFOLIO_2XA10_POOL (or a local dry run on one GPU).

argv: <fairsched tar> <model tar> <plan> [budget minutes] [kill-at seconds for the sweep's kill run]

Fetches the assets from @PORTFOLIO_LAB.GPU.ASSETS (PF_LOCAL_ASSETS=<dir> for a dry run), checks them against
SHA256SUMS, unpacks the CUDA 13.2 compat driver, the frozen vLLM 0.31.0 wheelhouse plus the ninja wheel, the
CUDA 13.2 nvcc copy (FlashInfer/Triton JIT), the model and the fairsched source tar (source at one commit plus
the static gateway and load-generator binaries), installs a venv with --no-index --no-deps, then runs the
plan's experiments one after another with fairsched's bench.replicas (one engine per GPU, the gateway, the
counter service, the load generator). After each run it prints a RESULT line with the run's key numbers and
PUTs the run directory (tar.gz) to @PORTFOLIO_LAB.GPU.JOBS/fairsched-results/<label>/. A run is not started
when the longest run so far would overrun the time budget; a failed run stops the plan.

Dry run: PF_LOCAL_ASSETS=<dir> PF_WORK=<dir on disk> PF_GPU_LOCK=/var/tmp/rtx4090.lock (each run holds the lock
for its own duration). A dry run keeps the venv between invocations (PF_WORK/venv/.installed).
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time

WORK = os.environ.get("PF_WORK", "/tmp/pf-fairsched")
LOCAL = os.environ.get("PF_LOCAL_ASSETS")
GPU_LOCK = os.environ.get("PF_GPU_LOCK")
NINJA = "ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
BASE = ["cuda-compat-13-2.tar", "wheelhouse-vllm-0.31.0.tar", NINJA, "cuda-13.2-nvcc.tar"]
VENV = os.path.join(WORK, "venv")
RESULTS_STAGE = "@PORTFOLIO_LAB.GPU.JOBS/fairsched-results"
OUTBOX = os.path.join(WORK, "outbox")
SERVED = "Qwen/Qwen2.5-7B-Instruct"
START = time.time()

ARMS = ["fcfs-rr", "vtc-rr", "gvtc-rr", "fcfs-sticky", "vtc-sticky", "gvtc-sticky"]


def rotated(seed):
    k = (seed - 1) % len(ARMS)
    return ARMS[k:] + ARMS[:k]


# Each entry: (workload, arm, seed, extra bench.replicas arguments).
PLANS = {
    # Two small engines on one GPU, 0.40 of its memory each.
    "dev": [("aggressive", "vtc-sticky", 1, []), ("aggressive", "gvtc-sticky", 1, []),
            ("aggressive", "gvtc-sticky", 1, ["--variant", "kill", "--kill-counters-at", "8"])],
    "dev-burst": [("noisy_neighbor", "gvtc-sticky", 1, ["--variant", "kill", "--kill-counters-at", "40"]),
                  ("noisy_neighbor", "fcfs-sticky", 1, [])],
    "aggressive": [("aggressive", arm, seed, []) for seed in (1, 2, 3) for arm in rotated(seed)],
    "burst": [("noisy_neighbor", arm, seed, []) for seed in (1, 2, 3) for arm in rotated(seed)],
    "sweep": [("aggressive", "gvtc-sticky", 1, ["--variant", "t10", "--sync-ms", "10"]),
              ("aggressive", "gvtc-sticky", 1, ["--variant", "t1000", "--sync-ms", "1000"]),
              ("aggressive", "gvtc-sticky", 1, ["--variant", "kill", "--kill-counters-at", "KILL_AT"])],
    # T = 10 ms again after the counter service got TCP_NODELAY, with T = 100 ms on the same node.
    "nodelay": [("aggressive", "gvtc-sticky", 1, ["--variant", "t10-nodelay", "--sync-ms", "10"]),
                ("aggressive", "gvtc-sticky", 1, ["--variant", "t100-nodelay", "--sync-ms", "100"])],
}

TORCH_CHECK = r"""
import json, torch
r = {"torch": torch.__version__, "cuda": torch.version.cuda, "available": torch.cuda.is_available()}
if r["available"]:
    r["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    x = torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
    r["matmul_finite"] = bool(torch.isfinite(x @ x).all())
print(json.dumps(r))
"""

SUMMARY = r"""
import json, sys
from pathlib import Path
from bench.replica_report import summarise_run
s = summarise_run(Path(sys.argv[1]))
a = s["client"]["tenants"].get("A", {})
out = {"run": s["run"], "workload": s["workload"], "arm": s["arm"], "seed": s["seed"], "variant": s["variant"],
       "startup_s": s["engine_startup_s"], "loadgen_rc": s["loadgen_returncode"], "routing": s["routing"],
       "share": (s["global_backlogged"] or {}).get("share"),
       "share_per_replica": (s["global_backlogged"] or {}).get("per_replica_share"),
       "backlogged_s": (s["global_backlogged"] or {}).get("seconds"),
       "ttft_A_ms": a.get("ttft_ms"), "completed": {t: v["completed"] for t, v in s["client"]["tenants"].items()},
       "tokens_per_s": round(s["client"]["output_tokens_per_s"], 1), "makespan_s": round(s["client"]["makespan_s"], 1),
       "sched_us_p50": [r and r["schedule_us"]["p50"] for r in s["replicas"]],
       "preemptions": [r and r["preemptions"] for r in s["replicas"]],
       "sync": [x and {"apply_us": x["apply_us"], "rtt_us": x["rtt_us"], "ok": x["ok"], "failed": x["failed"],
                       "events": x["events"]} for x in s["sync"]]}
for k in ("counters_killed_s", "global_backlogged_before_kill", "global_backlogged_after_kill"):
    if k in s:
        out[k] = s[k]
print(json.dumps(out))
"""


def emit(step, value):
    print("RESULT " + json.dumps({"step": step, step: value}), flush=True)


def sh(cmd, env=None, timeout=900, cwd=None):
    """Run cmd in its own process group; on timeout kill the whole group and keep the output so far."""
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=cwd,
                             start_new_session=True)
    except Exception as e:
        return {"rc": None, "out": "", "err": repr(e)[:500]}
    try:
        out, err = p.communicate(timeout=timeout)
        return {"rc": p.returncode, "out": out, "err": err}
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, err = p.communicate()
        return {"rc": None, "out": out, "err": (err or "") + f"\nTIMEOUT after {timeout} s: process group killed"}


def last_json(res):
    for line in reversed((res.get("out") or "").splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return {"rc": res.get("rc"), "err_tail": (res.get("err") or "").strip().splitlines()[-12:]}


def text(cmd, timeout=60):
    r = sh(cmd, timeout=timeout)
    return (r["out"] or r["err"] or "").strip()[:3000]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def untar(path, dest):
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(path) as t:
        try:
            t.extractall(dest, filter="data")
        except TypeError:
            t.extractall(dest)


def fetch(names):
    """Local path of each asset: the dry-run directory, or a GET from the stage into WORK."""
    os.makedirs(WORK, exist_ok=True)
    if LOCAL:
        return {n: os.path.join(LOCAL, n) for n in names + ["SHA256SUMS"]}
    from snowflake.snowpark import Session
    s = Session.builder.getOrCreate()
    for n in names + ["SHA256SUMS"]:
        s.file.get(f"@PORTFOLIO_LAB.GPU.ASSETS/{n}", WORK)
    return {n: os.path.join(WORK, n) for n in names + ["SHA256SUMS"]}


def bootstrap(src_name, model_name, out):
    reuse = bool(LOCAL) and os.path.exists(os.path.join(VENV, ".installed"))
    names = ([] if reuse else BASE) + [src_name, model_name]
    t = time.time()
    paths = fetch(names)
    out["fetch_s"] = round(time.time() - t, 1)
    want = dict(reversed(line.split()) for line in open(paths["SHA256SUMS"]) if line.strip())
    t = time.time()
    out["sha256_mismatch"] = [n for n in names if sha256(paths[n]) != want.get(n)]
    out["sha256_s"] = round(time.time() - t, 1)
    if out["sha256_mismatch"]:
        raise RuntimeError(f"sha256 mismatch: {out['sha256_mismatch']}")
    compat, wheels, src, model = (os.path.join(WORK, d) for d in ("compat", "wheels", "src", "model"))
    for d in (src, model):
        shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    unpack = [(src, src_name), (model, model_name)]
    if not reuse:
        unpack += [(compat, BASE[0]), (wheels, BASE[1]), (WORK, BASE[3])]
    for dest, n in unpack:
        untar(paths[n], dest)
        if not LOCAL:
            os.remove(paths[n])
    out["untar_s"] = round(time.time() - t, 1)
    if not reuse:
        t = time.time()
        shutil.rmtree(VENV, ignore_errors=True)
        sh([sys.executable, "-m", "venv", VENV], timeout=300)
        whls = sorted(os.path.join(wheels, f) for f in os.listdir(wheels) if f.endswith(".whl"))
        if NINJA not in map(os.path.basename, whls):
            whls.append(paths[NINJA])
        r = sh([os.path.join(VENV, "bin", "pip"), "install", "-q", "--no-index", "--no-deps", *whls], timeout=1800)
        out["install"] = {"rc": r["rc"], "wheels": len(whls), "seconds": round(time.time() - t, 1),
                          "err_tail": (r.get("err") or "").strip().splitlines()[-4:]}
        shutil.rmtree(wheels, ignore_errors=True)
        if r["rc"] != 0:
            raise RuntimeError("venv install failed")
        open(os.path.join(VENV, ".installed"), "w").close()
    out["venv_reused"] = reuse
    py = os.path.join(VENV, "bin", "python")
    cuda_home = os.path.join(WORK, "cuda-13.2")
    caches = os.path.join(WORK, "cache")
    for name in ("HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "PYTHONPATH"):
        os.environ.pop(name, None)
    base = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                XDG_CACHE_HOME=caches, VLLM_CACHE_ROOT=os.path.join(caches, "vllm"),
                FLASHINFER_WORKSPACE_BASE=os.path.join(caches, "flashinfer"),
                TRITON_CACHE_DIR=os.path.join(caches, "triton"), CUDA_HOME=cuda_home,
                TRITON_PTXAS_PATH=os.path.join(cuda_home, "bin", "ptxas"),
                PATH=os.path.join(VENV, "bin") + ":" + os.path.join(cuda_home, "bin") + ":" + os.environ["PATH"])
    lock = ["flock", GPU_LOCK] if GPU_LOCK else []
    node = last_json(sh(lock + [py, "-c", TORCH_CHECK], env=base, timeout=300))
    env = base
    if not node.get("available"):
        env = dict(base, LD_LIBRARY_PATH=compat + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        out["torch_compat"] = last_json(sh(lock + [py, "-c", TORCH_CHECK], env=env, timeout=300))
    out["torch_node_driver"] = node
    out["compat_driver_used"] = env is not base
    repo = os.path.join(src, "fairsched")
    env["FAIRSCHED_SOURCE_COMMIT"] = open(os.path.join(repo, "SOURCE_COMMIT")).read().strip()
    out["source_commit"] = env["FAIRSCHED_SOURCE_COMMIT"]
    model_dir = next(root for root, _, files in os.walk(model) if "config.json" in files)
    return py, env, repo, model_dir


def put_results(label):
    if LOCAL or not os.path.isdir(OUTBOX):
        return {"outbox": OUTBOX}
    from snowflake.snowpark import Session
    s = Session.builder.getOrCreate()
    sent = {}
    for name in sorted(os.listdir(OUTBOX)):
        if name.endswith(".sent"):
            continue
        try:
            r = s.file.put(os.path.join(OUTBOX, name), f"{RESULTS_STAGE}/{label}/", auto_compress=False,
                           overwrite=True)[0]
            sent[name] = r.status
            os.rename(os.path.join(OUTBOX, name), os.path.join(OUTBOX, name + ".sent"))
        except Exception as e:
            sent[name] = repr(e)[:200]
    return sent


def key_lines(path, limit=25):
    words = ("Error", "error:", "Traceback", "out of memory", "OutOfMemory", "CUDA", "RuntimeError", "Killed")
    try:
        lines = open(path, errors="replace").read().splitlines()
    except OSError:
        return []
    return [x.strip()[:300] for x in lines if any(w in x for w in words)][-limit:] or \
        [x.strip()[:300] for x in lines[-8:]]


try:
    src_name, model_name, plan = sys.argv[1], sys.argv[2], sys.argv[3]
    budget = float(sys.argv[4]) * 60 if len(sys.argv) > 4 else 50 * 60
    label = os.environ.get("PF_RESULTS_LABEL", plan.replace(",", "+"))
    kill_at = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("PF_KILL_AT", "30")
    emit("node", {"python": sys.version.split()[0], "cpus": os.cpu_count(), "gpus": text(["nvidia-smi", "-L"]),
                  "topo": text(["nvidia-smi", "topo", "-m"]),
                  "query": text(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used",
                                 "--format=csv,noheader"])})
    boot = {}
    try:
        py, env, repo, model_dir = bootstrap(src_name, model_name, boot)
    finally:
        emit("bootstrap", boot)
    dry = bool(LOCAL)
    served = "Qwen/Qwen2.5-0.5B-Instruct" if "0.5b" in model_name else SERVED
    common = ["--model", model_dir, "--served-model", served, "--vllm", os.path.join(VENV, "bin", "vllm"),
              "--proxy", os.path.join(repo, "bin", "fairsched-proxy"),
              "--loadgen", os.path.join(repo, "bin", "fairsched-loadgen")]
    if dry:
        common += ["--gpus", "0,0", "--gpu-memory-utilization", "0.40", "--proxy-port", "28080",
                   "--counter-port", "28090", "--engine-port", "28001"]
        if GPU_LOCK:
            common += ["--gpu-lock", GPU_LOCK]
    else:
        # A fixed KV size (4.5 GiB, 84k tokens, as on the RTX 4090): with 0.90 and the profiled size, every
        # run after the first loaded the torch.compile cache, profiled 0.77 GiB more KV than the fresh
        # compile, and ran out of memory at its first requests (job 20261105T070843Z-fairsched-aggressive).
        common += ["--gpus", "0,1", "--gpu-memory-utilization", "0.88",
                   "--engine-arg=--kv-cache-memory-bytes=4831838208"]
    longest = 0.0
    for workload, arm, seed, extra in [e for p in plan.split(",") for e in PLANS[p]]:  # "burst,sweep"
        extra = [kill_at if x == "KILL_AT" else x for x in extra]
        name = f"{workload}/{arm}{'-' + extra[1] if extra[:1] == ['--variant'] else ''}-s{seed}"
        if time.time() - START + longest > budget:
            emit(name, {"skipped": f"would overrun the {budget / 60:.0f}-minute budget"})
            continue
        out_dir = os.path.join(WORK, "runs", workload)
        t = time.time()
        r = sh([py, "-m", "bench.replicas", workload, arm, "--seed", str(seed), "--out", out_dir, *common, *extra],
               env=env, timeout=1800, cwd=repo)
        seconds = round(time.time() - t, 1)
        longest = max(longest, seconds)
        run_dir = (r["out"] or "").strip().splitlines()[-1] if r["rc"] == 0 else None
        value = {"rc": r["rc"], "seconds": seconds}
        if run_dir and os.path.isdir(run_dir):
            summary = sh([py, "-c", SUMMARY, run_dir], env=env, timeout=300, cwd=repo)
            value["summary"] = last_json(summary)
            os.makedirs(OUTBOX, exist_ok=True)
            archive = os.path.join(OUTBOX, f"{workload}--{os.path.basename(run_dir)}.tar.gz")
            with tarfile.open(archive, "w:gz") as tar:
                tar.add(run_dir, arcname=f"{workload}/{os.path.basename(run_dir)}")
            value["archive_bytes"] = os.path.getsize(archive)
            done = (value["summary"].get("completed") or {})
            if value["summary"].get("loadgen_rc") != 0 or not done or not all(done.values()):
                value["rc"] = r["rc"] = "no_completions"   # a dead engine fails every request quickly: stop the plan
        else:
            value["err_tail"] = (r["err"] or "").strip().splitlines()[-15:]
            failed = os.path.join(out_dir, f"{arm}{'-' + extra[1] if extra[:1] == ['--variant'] else ''}-s{seed}")
            for log in ("engine0.log", "engine1.log", "counters.log", "proxy.log", "loadgen.out"):
                value[log] = key_lines(os.path.join(failed, log))
        emit(name, value)
        emit(name + " put", put_results(label))
        if r["rc"] != 0:
            emit("stopped", f"{name} failed: the rest of the plan is skipped")
            break
except Exception:
    import traceback
    emit("error", traceback.format_exc()[-2000:])
emit("elapsed_s", round(time.time() - START, 1))
