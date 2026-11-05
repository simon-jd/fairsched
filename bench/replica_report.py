"""Summarise multi-replica runs (``bench.replicas``): global service share, latency, sync cost.

    python3 -m bench.replica_report results/raw/replicas --out results/replicas

Writes ``<out>/summary.json``: one entry per run directory, and per
(workload, arm) the range over seeds of the measures the README quotes.
With ``--compact``, also copies each run's raw records into
``<out>/runs/<run>/`` in compressed form (what the summary is computed from).

The global share is each tenant's share of billed service, summed over the
replicas, over the time when every tenant had requests waiting in some
engine's queue (fair = 0.50 for two equal-weight tenants).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import shutil
import statistics

from bench.report import client_summary, load_client, load_steps, percentile, step_summary

TENANTS = ("A", "B")


def replica_steps(run: Path) -> list[list[dict]]:
    dirs = sorted((p for p in run.glob("replica*") if p.is_dir()), key=lambda p: int(p.name[7:]))
    return [load_steps(d) for d in dirs]


def global_share(replicas: list[list[dict]], tenants=TENANTS, since: float | None = None,
                 until: float | None = None) -> dict | None:
    """Billed service per tenant, summed over replicas, while every tenant waits somewhere.

    Steps from all replicas are merged in time order. A replica's step counts
    when, just before it, every tenant had a request waiting on some replica
    (each replica's waiting counts as of its last step). ``since``/``until``
    bound the steps counted (Unix seconds).
    """
    events = sorted(((s["t"], i, s) for i, steps in enumerate(replicas) for s in steps if s.get("type") == "step"),
                    key=lambda e: (e[0], e[1]))
    waiting = [dict() for _ in replicas]
    accounts = [dict() for _ in replicas]
    totals = {t: 0.0 for t in tenants}
    per_replica = [{t: 0.0 for t in tenants} for _ in replicas]
    seconds, last_t = 0.0, None
    for t, i, step in events:
        backlogged = all(sum(w.get(tenant, 0) for w in waiting) > 0 for tenant in tenants)
        counted = backlogged and (since is None or t >= since) and (until is None or t < until)
        if counted and accounts[i]:
            for tenant in tenants:
                delta = step["accounts"].get(tenant, 0) - accounts[i].get(tenant, 0)
                totals[tenant] += delta
                per_replica[i][tenant] += delta
        if counted and last_t is not None:
            seconds += t - last_t
        last_t = t
        waiting[i] = step["waiting"]
        accounts[i] = step["accounts"]
    total = sum(totals.values())
    if not total:
        return None
    return {"share": {t: round(v / total, 4) for t, v in totals.items()},
            "billed": {t: round(v, 1) for t, v in totals.items()},
            "per_replica_share": [{t: round(v / sum(r.values()), 4) for t, v in r.items()} if sum(r.values()) else None
                                  for r in per_replica],
            "seconds": round(seconds, 1)}


def routing(run: Path) -> dict[str, dict[str, int]]:
    """Requests per tenant and upstream, from the gateway's terminal records."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    path = run / "proxy.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "request_terminal" and rec.get("tenant"):
                out[rec["tenant"]][rec.get("upstream", "")] += 1
    return {t: dict(sorted(v.items())) for t, v in sorted(out.items())}


def sync_summary(steps: list[dict], since: float | None) -> dict | None:
    rows = [s for s in steps if s.get("type") == "step" and s.get("sync_ns") is not None
            and (since is None or s["t"] >= since)]
    events = [{k: e[k] for k in ("state", "t", "error") if k in e} for e in steps if e.get("type") == "counter_sync"]
    end = next((e for e in steps if e.get("type") == "counter_sync_end"), None)
    if not rows and not events:
        return None
    sync_us = [s["sync_ns"] / 1e3 for s in rows]
    last = rows[-1] if rows else {}
    rtt = [s["sync"]["rtt_us"] for s in rows if s.get("sync", {}).get("rtt_us") is not None]
    return {"steps": len(rows),
            "rtt_us": (end or {}).get("rtt_us") or {"p50": percentile(rtt, 50), "p99": percentile(rtt, 99),
                                                   "from": "last round trip at each step"},
            "apply_us": {"p50": percentile(sync_us, 50), "p99": percentile(sync_us, 99),
                         "mean": round(statistics.fmean(sync_us), 3) if sync_us else None},
            "ok": (end or last.get("sync", {})).get("ok"), "failed": (end or last.get("sync", {})).get("failed"),
            "events": events, "remote_final": last.get("remote")}


def summarise_run(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text())
    since = meta.get("trace_start_unix")
    replicas = replica_steps(run)
    out = {"run": run.name, "workload": meta["workload"], "arm": meta["arm"], "seed": meta["seed"],
           "variant": meta.get("variant"), "sync_ms": meta.get("sync_ms"), "git_commit": meta.get("git_commit"),
           "engine_startup_s": meta.get("engine_startup_s"), "loadgen_returncode": meta.get("loadgen_returncode"),
           "routing": routing(run)}
    out["client"] = client_summary(load_client(run / "client.jsonl"))
    out["global_backlogged"] = global_share(replicas, since=since)
    out["replicas"] = [step_summary(steps, since) for steps in replicas]
    out["sync"] = [sync_summary(steps, since) for steps in replicas]
    killed = meta.get("counters_killed_unix")
    if killed:
        out["counters_killed_s"] = round(killed - since, 1)
        out["global_backlogged_before_kill"] = global_share(replicas, since=since, until=killed)
        out["global_backlogged_after_kill"] = global_share(replicas, since=killed)
    return out


def _range(values: list[float | None]) -> list[float] | None:
    values = [v for v in values if v is not None]
    return [round(min(values), 4), round(max(values), 4)] if values else None


def table(runs: list[dict]) -> dict:
    """Per workload and arm (and variant): ranges over seeds of the quoted measures."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        groups[f"{r['workload']}/{r['arm']}{'-' + r['variant'] if r['variant'] else ''}"].append(r)
    out = {}
    for key, rows in sorted(groups.items()):
        a = [r["client"]["tenants"].get("A", {}) for r in rows]
        sched = [s["schedule_us"]["p50"] for r in rows for s in r["replicas"] if s]
        apply_us = [s["apply_us"]["p50"] for r in rows for s in r["sync"] if s and s["apply_us"]["p50"] is not None]
        out[key] = {
            "seeds": sorted(r["seed"] for r in rows),
            "share_A": _range([(r["global_backlogged"] or {}).get("share", {}).get("A") for r in rows]),
            "ttft_A_p50_ms": _range([t.get("ttft_ms", {}).get("p50") for t in a]),
            "ttft_A_p99_ms": _range([t.get("ttft_ms", {}).get("p99") for t in a]),
            "output_tokens_per_s": _range([r["client"]["output_tokens_per_s"] for r in rows]),
            "output_tokens_per_s_median": statistics.median(r["client"]["output_tokens_per_s"] for r in rows),
            "schedule_us_p50": _range(sched),
            "sync_apply_us_p50": _range(apply_us),
            "preemptions": [sum(sum((s or {}).get("preemptions", {}).values()) for s in r["replicas"]) for r in rows],
        }
    return out


def compact(run: Path, dest: Path) -> None:
    """The raw records a summary comes from, compressed: step logs, per-request client and gateway records."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("meta.json", "client-report.json", "policy.json", "proxy.json"):
        if (run / name).exists():
            shutil.copy(run / name, dest / name)
    sessions = {}
    for line in (run / "trace.jsonl").read_text().splitlines():
        row = json.loads(line)
        sessions[row["id"]] = row.get("session")
    upstream = {}
    if (run / "proxy.jsonl").exists():
        for line in (run / "proxy.jsonl").read_text().splitlines():
            rec = json.loads(line)
            if rec.get("event") == "request_terminal":
                upstream[rec["rid"]] = rec.get("upstream")
    keep = ("id", "rid", "tenant", "scheduled_unix_ns", "sent_unix_ns", "first_output_unix_ns", "last_output_unix_ns",
            "finished_unix_ns", "state", "status", "reason", "output_chunks", "completion_tokens")
    with gzip.open(dest / "requests.jsonl.gz", "wt") as f:
        for row in load_client(run / "client.jsonl"):
            rec = {k: row[k] for k in keep if k in row}
            rec["session"] = sessions.get(row["id"])
            rec["upstream"] = upstream.get(row.get("rid"))
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    for i, d in enumerate(sorted((p for p in run.glob("replica*") if p.is_dir()), key=lambda p: int(p.name[7:]))):
        with gzip.open(dest / f"steps-r{i}.jsonl.gz", "wt") as f:
            for path in sorted(d.glob("steps-*.jsonl")):
                f.write(path.read_text())
    if (run / "gpu.csv").exists():
        with gzip.open(dest / "gpu.csv.gz", "wt") as f:
            f.write((run / "gpu.csv").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("raw", nargs="+", help="directories holding run directories")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    runs = sorted(p for raw in args.raw for p in Path(raw).iterdir()
                  if (p / "meta.json").exists() and (p / "client.jsonl").exists())
    summaries = [summarise_run(run) for run in runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({"table": table(summaries), "runs": summaries}, indent=1) + "\n")
    if args.compact:
        for run, summary in zip(runs, summaries):
            compact(run, out / "runs" / summary["workload"] / run.name)
    print(out / "summary.json")


if __name__ == "__main__":
    main()
