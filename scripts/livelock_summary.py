"""Summarise the step log of the earlier quantum-preemption design's livelocked run.

    python3 scripts/livelock_summary.py <run dir> > results/livelock/summary.json

<run dir> is the earlier design's run directory, which held its frozen source
(source/), run plan and step log; none of it is in this repository. Counts
come from the log's per-step records; hashes identify the log and the source.
"""
import collections
import glob
import gzip
import hashlib
import json
import os
import sys

run_dir = sys.argv[1]
engine = glob.glob(os.path.join(run_dir, "results", "lifecycle", "*", "engine"))[0]
plan = json.load(open(os.path.join(run_dir, "plan.json")))
design = sorted(glob.glob(os.path.join(run_dir, "source", "p02_sched", "*.py")))
shards = sorted(glob.glob(os.path.join(engine, "steps", "*.jsonl.gz")))
pages = sorted(glob.glob(os.path.join(engine, "steps", "page-*.json")))
groups = sorted(glob.glob(os.path.join(engine, "steps", "group-*.json")))


def find(obj, key):
    """First value stored under ``key`` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        obj = list(obj.values())
    if isinstance(obj, list):
        for value in obj:
            found = find(value, key)
            if found is not None:
                return found
    return None


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


manifest = "".join(f"{sha(p)}  steps/{os.path.basename(p)}\n" for p in shards + pages + groups)
steps = 0
forced = 0
native_preempted = 0
per_req = collections.defaultdict(lambda: collections.Counter())
tenant_of = {}
terminal = []
header = None
ts = []
step_seqs = []
for path in shards:
    for line in gzip.open(path, "rt"):
        r = json.loads(line)
        d = r.get("data", {})
        if r.get("kind") != "engine_event":
            continue
        if header is None and "policy" in d and "num_blocks" in d:
            header = d
        t = d.get("type")
        if t == "step":
            steps += 1
            ts.append(d["timestamp_unix_ms"])
            step_seqs.append(d["step_seq"])
            forced += len(d["decision"].get("forced_preemptions", []))
            native_preempted += len(d["decision"].get("preempted", []))
            for s in d.get("scheduled", []):
                c = per_req[s["request_id"]]
                c["prefill"] += s["prefill_tokens"]
                c["decode"] += s["decode_tokens"]
                c["recompute_prefill"] += s["recompute_prefill_tokens"]
                c["steps"] += 1
                tenant_of[s["request_id"]] = s["tenant"]
        elif t == "engine_terminal":
            terminal.append((d["request_id"], d["tenant"], d["status"], d["output_tokens"], d["step_seq"]))

index = json.load(open(os.path.join(engine, "steps-index.json")))
prefill = sum(c["prefill"] for c in per_req.values())
recompute = sum(c["recompute_prefill"] for c in per_req.values())
out = {
    "source": "step log of the earlier quantum-preemption design (not in this repository)",
    "design": {
        "scheduler_py_sha256": sha(os.path.join(run_dir, "source", "p02_sched", "scheduler.py")),
        "package_manifest_sha256": hashlib.sha256("".join(
            f"{sha(p)}  {os.path.basename(p)}\n" for p in design).encode()).hexdigest(),
        "package_files": [os.path.basename(p) for p in design],
    },
    "run": {"model": find(plan, "model"), "max_num_batched_tokens": find(plan, "max_num_batched_tokens"),
            "max_model_len": find(plan, "max_model_len"), "prefix_caching": find(plan, "prefix_caching"),
            "client_timeout_ms": find(plan, "row_timeout_ms")},
    "policy_version": header["policy_version"] if header and "policy_version" in header else header["policy"].get("version"),
    "policy": header["policy"],
    "num_blocks": header["num_blocks"],
    "step_log": {
        "files": len(shards) + len(pages) + len(groups),
        "manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
        "index_sha256": sha(os.path.join(engine, "steps-index.json")),
        "footer_rows": index["footer"]["rows"],
        "footer_closed": index["footer"]["closed"],
        "footer_first_error": index["footer"].get("first_error") is not None,
    },
    "steps": steps,
    "step_seq_range": [min(step_seqs), max(step_seqs)],
    "steps_span_s": round((max(ts) - min(ts)) / 1000, 1),
    "policy_preemptions": forced,
    "native_preemptions": native_preempted,
    "requests_scheduled": len(per_req),
    "per_request": {k: {"tenant": tenant_of[k], **dict(v)} for k, v in sorted(per_req.items())},
    "prefill_tokens": prefill,
    "recompute_prefill_tokens": recompute,
    "recompute_fraction": round(recompute / prefill, 4) if prefill else None,
    "decode_tokens": sum(c["decode"] for c in per_req.values()),
    "terminal": [dict(zip(("request_id", "tenant", "status", "output_tokens", "step_seq"), t)) for t in sorted(terminal)],
}
print(json.dumps(out, indent=1))
