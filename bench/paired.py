"""Compare arms seed for seed on throughput, with the run order taken into account.

    python3 -m bench.paired overhead_rotated fcfs:native vtc:fcfs vtc:native
    python3 -m bench.paired kv_victim_rotated vtc-kv:vtc

Reads ``results/<name>/summary.json`` (written by bench/report.py), in which
every seed ran each arm once on the same trace, and writes
``results/<name>/paired.json``. For each ``a:b`` pair it gives the per-seed
difference in output tokens/s, ``100 * (a / b - 1)`` percent, the mean over
seeds and a two-sided 95% Student t interval for that mean. A difference is
marked resolved only when the interval excludes zero.

The runs of a seed are numbered 1, 2, ... by start time. ``position_effect_pct``
is, for each position, the mean throughput of the runs in it relative to the
mean of their seed's runs, so a run-order effect shows up there. Each arm's
per-seed median SM clock and peak GPU temperature are copied alongside,
because clocks could not be locked. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

# Two-sided 95% quantiles of Student's t, by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def throughput(run: dict) -> float:
    return run["client"]["output_tokens_per_s"]


def mean_interval(values: list[float]) -> tuple[float, float, float]:
    """Mean and two-sided 95% t interval of the mean."""
    df = len(values) - 1
    if df not in T95:
        raise ValueError(f"need 2 to {max(T95) + 1} values, got {len(values)}")
    m = statistics.fmean(values)
    half = T95[df] * statistics.stdev(values) / len(values) ** 0.5
    return m, m - half, m + half


def analyse_runs(runs: list[dict], pairs: list[tuple[str, str]]) -> dict:
    by = {(r["arm"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    arms = sorted({r["arm"] for r in runs})
    missing = [(a, s) for a in arms for s in seeds if (a, s) not in by]
    if missing:
        raise ValueError(f"not every arm ran every seed: missing {missing}")
    position = {}
    for s in seeds:
        ordered = sorted((r for r in runs if r["seed"] == s), key=lambda r: r["started_unix"])
        for i, r in enumerate(ordered, 1):
            position[(r["arm"], s)] = i

    out_arms = {}
    for arm in arms:
        tps = [throughput(by[(arm, s)]) for s in seeds]
        med = statistics.median(tps)
        gpu = [by[(arm, s)].get("gpu", {}) for s in seeds]
        out_arms[arm] = {
            "output_tokens_per_s": [round(t, 2) for t in tps],
            "median": round(med, 2),
            "spread_pct": round(100 * (max(tps) - min(tps)) / med, 3),
            "position": [position[(arm, s)] for s in seeds],
            "sm_clock_mhz_p50": [g.get("sm_clock_mhz", {}).get("p50") for g in gpu],
            "temperature_c_max": [g.get("temperature_c", {}).get("max") for g in gpu],
        }

    relative = {}
    for (arm, s), i in position.items():
        seed_mean = statistics.fmean(throughput(by[(a, s)]) for a in arms)
        relative.setdefault(i, []).append(100 * (throughput(by[(arm, s)]) / seed_mean - 1))

    out_pairs = []
    for a, b in pairs:
        if a not in arms or b not in arms:
            raise ValueError(f"unknown arm in {a}:{b}; arms are {arms}")
        diffs = [100 * (throughput(by[(a, s)]) / throughput(by[(b, s)]) - 1) for s in seeds]
        m, lo, hi = mean_interval(diffs)
        out_pairs.append({
            "a": a, "b": b,
            "diff_pct": [round(d, 3) for d in diffs],
            "mean_pct": round(m, 3),
            "ci95_pct": [round(lo, 3), round(hi, 3)],
            "resolved": lo > 0 or hi < 0,
            "a_ran_first": [position[(a, s)] < position[(b, s)] for s in seeds],
        })

    return {
        "metric": "client.output_tokens_per_s",
        "difference": "100 * (a / b - 1) per seed, percent",
        "interval": "two-sided 95% Student t interval of the mean difference over seeds; "
                    "resolved means the interval excludes zero",
        "seeds": seeds,
        "arms": out_arms,
        "position_effect_pct": {str(i): round(statistics.fmean(v), 3) for i, v in sorted(relative.items())},
        "pairs": out_pairs,
    }


def analyse(name: str, pairs: list[tuple[str, str]]) -> Path:
    summary = json.loads((ROOT / "results" / name / "summary.json").read_text())
    out = {"workload": summary["workload"], "source": f"results/{name}/summary.json",
           **analyse_runs(summary["runs"], pairs)}
    path = ROOT / "results" / name / "paired.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="directory under results/ holding summary.json")
    parser.add_argument("pairs", nargs="+", metavar="A:B", help="compare arm A with arm B")
    args = parser.parse_args()
    pairs = []
    for p in args.pairs:
        a, sep, b = p.partition(":")
        if not sep or not a or not b:
            parser.error(f"pair {p!r} is not A:B")
        pairs.append((a, b))
    print(analyse(args.name, pairs))


if __name__ == "__main__":
    main()
