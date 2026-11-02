"""Summarise experiment runs: per-tenant latency, service share and scheduler cost.

    python3 -m bench.report noisy_neighbor          # all runs under results/raw/noisy_neighbor
    python3 -m bench.report noisy_neighbor --plots  # also write PNGs (needs matplotlib)
    python3 -m bench.report overhead                # runs made with bench.run --name overhead

Writes ``results/<name>/summary.json`` (committed) from the raw run
directories under ``results/raw/<name>/`` (not committed). Latency is
client-observed: TTFT is first output chunk minus the request's scheduled
arrival time, so queueing in the gateway and engine is included.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def load_client(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_steps(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("steps-*.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break       # last line cut off by a killed engine
    return rows


def client_summary(results: list[dict]) -> dict:
    by_tenant: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_tenant[r["tenant"]].append(r)
    out = {}
    start = min(r["scheduled_unix_ns"] for r in results)
    end = max(r["finished_unix_ns"] for r in results)
    for tenant, rows in sorted(by_tenant.items()):
        done = [r for r in rows if r["state"] == "completed"]
        ttft = [(r["first_output_unix_ns"] - r["scheduled_unix_ns"]) / 1e6 for r in done if r.get("first_output_unix_ns")]
        e2e = [(r["finished_unix_ns"] - r["scheduled_unix_ns"]) / 1e6 for r in done]
        tpot = [(r["last_output_unix_ns"] - r["first_output_unix_ns"]) / 1e6 / (r["completion_tokens"] - 1)
                for r in done if r.get("completion_tokens", 0) and r["completion_tokens"] > 1]
        tokens = sum(r.get("completion_tokens") or 0 for r in done)
        finish = max((r["finished_unix_ns"] for r in done), default=start)
        out[tenant] = {
            "requests": len(rows), "completed": len(done),
            "cancelled": sum(r["state"] == "cancelled" for r in rows),
            "failed": sum(r["state"] not in {"completed", "cancelled"} for r in rows),
            "ttft_ms": {"p50": percentile(ttft, 50), "p90": percentile(ttft, 90), "p99": percentile(ttft, 99)},
            "e2e_ms": {"p50": percentile(e2e, 50), "p99": percentile(e2e, 99)},
            "tpot_ms_p50": percentile(tpot, 50),
            "output_tokens": tokens,
            "last_finish_s": (finish - start) / 1e9,
        }
    total = sum(t["output_tokens"] for t in out.values())
    return {"tenants": out, "makespan_s": (end - start) / 1e9,
            "output_tokens_per_s": total / ((end - start) / 1e9)}


def step_summary(steps: list[dict], since: float | None = None) -> dict | None:
    rows = [s for s in steps if s.get("type") == "step" and (since is None or s["t"] >= since)]
    if not rows:
        return None
    tenants = sorted({t for s in rows for t in s["scheduled"]} | {t for s in rows for t in s["counters"]})
    sched_us = [s["sched_ns"] / 1e3 for s in rows]
    first, last = _baseline(steps, since), rows[-1]
    preempted = defaultdict(int)
    for s in rows:
        for tenant, n in s["preempted"].items():
            preempted[tenant] += n
    return {
        "steps": len(rows),
        "schedule_us": {"p50": percentile(sched_us, 50), "p99": percentile(sched_us, 99),
                        "mean": statistics.fmean(sched_us)},
        "billed": {t: last["accounts"].get(t, 0) - first["accounts"].get(t, 0) for t in [*tenants, "system"]},
        "wasted": {t: last["wasted"].get(t, 0) - first["wasted"].get(t, 0) for t in tenants},
        "preemptions": dict(preempted),
        "min_free_blocks": min(s["free_blocks"] for s in rows),
        "backlogged_share": backlogged_share(rows, [t for t in tenants if t != "default"]),
    }


def backlogged_share(rows: list[dict], tenants: list[str]) -> dict[str, float] | None:
    """Each tenant's share of billed service over steps where every tenant had work waiting.

    This is the regime VTC's fairness guarantee is about; outside it, an idle
    tenant's share is limited by its own demand.
    """
    totals = {t: 0.0 for t in tenants}
    for prev, step in zip(rows, rows[1:]):
        if tenants and all(prev["waiting"].get(t, 0) > 0 for t in tenants):
            for t in tenants:
                totals[t] += step["accounts"].get(t, 0) - prev["accounts"].get(t, 0)
    total = sum(totals.values())
    return {t: round(v / total, 4) for t, v in totals.items()} if total else None


def _baseline(steps: list[dict], since: float | None) -> dict:
    """Cumulative accounts just before ``since``: the last earlier step (warm-up), else zero."""
    earlier = [s for s in steps if s.get("type") == "step" and since is not None and s["t"] < since]
    return earlier[-1] if earlier else {"accounts": {}, "wasted": {}}


def service_timeline(steps: list[dict], since: float, bin_s: float = 2.0) -> dict[str, list[float]]:
    """Per-tenant billed cost per second, in ``bin_s`` bins from ``since``."""
    bins: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    prev = _baseline(steps, since)["accounts"]
    for s in steps:
        if s.get("type") != "step" or s["t"] < since:
            continue
        index = int((s["t"] - since) / bin_s)
        for tenant, value in s["accounts"].items():
            bins[tenant][index] += value - prev.get(tenant, 0)
        prev = s["accounts"]
    width = max((max(b) for b in bins.values() if b), default=-1) + 1
    return {t: [b.get(i, 0.0) / bin_s for i in range(width)] for t, b in sorted(bins.items())}


def gpu_summary(path: Path) -> dict | None:
    """SM clock, temperature and power over a run, from the runner's gpu.csv.

    Runs recorded before the runner sampled clocks have only three columns and
    get no summary.
    """
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) >= 8:
                try:
                    rows.append((float(fields[3]), float(fields[5]), float(fields[6]), fields[7]))
                except ValueError:
                    continue
    if not rows:
        return None
    sm, temp, power, pstate = zip(*rows)
    return {"samples": len(rows), "sm_clock_mhz": {"p50": percentile(sm, 50), "min": min(sm)},
            "temperature_c": {"p50": percentile(temp, 50), "max": max(temp)},
            "power_w": {"p50": percentile(power, 50)}, "pstates": sorted(set(pstate))}


def summarise_run(run: Path, previous: dict | None = None) -> dict:
    """One run's summary. ``source_tree`` comes from the run's meta.json; a run
    recorded before meta.json carried it keeps the value in ``previous``, the
    existing summary's ``{(arm, seed): source_tree}``."""
    meta = json.loads((run / "meta.json").read_text())
    tree = meta.get("source_tree") or (previous or {}).get((meta["arm"], meta["seed"]))
    summary = {"arm": meta["arm"], "seed": meta["seed"], "source_tree": tree}
    if meta.get("source_dirty"):
        summary["source_dirty"] = True
    if meta.get("trace_start_unix") is not None:
        summary["started_unix"] = round(meta["trace_start_unix"], 1)
    gpu = gpu_summary(run / "gpu.csv")
    if gpu is not None:
        summary["gpu"] = gpu
    summary["client"] = client_summary(load_client(run / "client.jsonl"))
    steps = load_steps(run)
    summary["engine"] = step_summary(steps, meta.get("trace_start_unix"))
    return summary


def summarise_workload(name: str, plots: bool = False) -> Path:
    raw = ROOT / "results/raw" / name
    runs = sorted(p for p in raw.iterdir() if (p / "meta.json").exists() and (p / "client.jsonl").exists())
    workload = json.loads((runs[0] / "meta.json").read_text())["workload"] if runs else name
    out = ROOT / "results" / name
    path = out / "summary.json"
    previous = {}
    if path.exists():
        previous = {(r["arm"], r["seed"]): r.get("source_tree") for r in json.loads(path.read_text())["runs"]}
    summaries = [summarise_run(run, previous) for run in runs]
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"workload": workload, "runs": summaries}, indent=1) + "\n")
    if plots:
        from bench.plots import plot_workload
        plot_workload(workload, runs, out)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="directories under results/raw (usually the workload)")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    for name in args.names:
        print(summarise_workload(name, args.plots))


if __name__ == "__main__":
    main()
