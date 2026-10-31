"""Fit step time to each step's GPU work and split GPU time between tenants.

    python3 -m bench.gpu_time mixed_shapes     # every run under results/raw/mixed_shapes

The step log records, per tenant and step, the decode steps ``d``, the query
tokens run as prefill ``q`` and the causal attention pairs ``attn``. With
async scheduling a step is logged while an earlier one is still on the GPU,
so step k's work is paired with the interval between log records k+offset and
k+offset+1. The offset that fits best (0 to 3) is used and reported. The model

    interval = c0 + cq * q + cd * d + ca * attn

is fitted by least squares over all runs of the set. Intervals over 1 s, which
include idle time, are left out, and a feature whose coefficient comes out
negative is dropped and the model refitted. Each interval is then split
between tenants in proportion to their predicted variable cost
(cq * q_t + cd * d_t + ca * attn_t), so the fixed cost c0 is shared in the
same proportion. A tenant's share of GPU time is reported over the steps in
which every tenant had requests waiting (the steps ``backlogged_share`` in
bench/report.py uses) and over the whole run, next to its share of billed
service over the same steps.

Writes ``results/<name>/gpu_time.json``. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from bench.report import load_steps

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("q", "d", "attn")
MAX_INTERVAL_S = 1.0
OFFSETS = (0, 1, 2, 3)
IGNORED = {"default", "system"}


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            raise ValueError("singular system: a feature is constant or collinear")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] / m[i][i] for i in range(n)]


def least_squares(rows: list[list[float]], y: list[float]) -> list[float]:
    """Ordinary least squares on columns scaled to unit maximum, for conditioning."""
    k = len(rows[0])
    scale = [max(abs(r[j]) for r in rows) or 1.0 for j in range(k)]
    x = [[r[j] / scale[j] for j in range(k)] for r in rows]
    xtx = [[sum(r[i] * r[j] for r in x) for j in range(k)] for i in range(k)]
    xty = [sum(r[i] * v for r, v in zip(x, y)) for i in range(k)]
    return [c / s for c, s in zip(_solve(xtx, xty), scale)]


def _totals(step: dict) -> dict[str, float]:
    work = step.get("work", {})
    return {f: float(sum(w.get(f, 0) for w in work.values())) for f in FEATURES}


def samples(steps: list[dict], offset: int) -> list[tuple[int, float]]:
    """(index of the step, interval in seconds) pairs used for fitting."""
    out = []
    for k in range(len(steps) - offset - 1):
        interval = steps[k + offset + 1]["t"] - steps[k + offset]["t"]
        if "work" in steps[k] and 0 < interval <= MAX_INTERVAL_S:
            out.append((k, interval))
    return out


def fit(runs: list[list[dict]], offset: int) -> dict:
    data = [(_totals(steps[k]), interval) for steps in runs for k, interval in samples(steps, offset)]
    features = list(FEATURES)
    dropped = []
    while True:
        rows = [[1.0] + [t[f] for f in features] for t, _ in data]
        y = [interval for _, interval in data]
        coef = least_squares(rows, y)
        negative = [(c, f) for c, f in zip(coef[1:], features) if c < 0]
        if not negative:
            break
        worst = min(negative)[1]
        features.remove(worst)
        dropped.append(worst)
    pred = [sum(c * x for c, x in zip(coef, row)) for row in rows]
    mean = statistics.fmean(y)
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred))
    return {"offset": offset, "intercept_s": coef[0], "coef_s": dict(zip(features, coef[1:])),
            "dropped": dropped, "r2": 1 - ss_res / ss_tot if ss_tot else None,
            "median_abs_error_s": statistics.median(abs(v - p) for v, p in zip(y, pred)), "intervals": len(y)}


def attribute(steps: list[dict], model: dict) -> dict:
    """Per-tenant GPU seconds and billed service, over backlogged steps and the whole run."""
    coef = model["coef_s"]
    tenants = sorted({t for s in steps for t in s.get("work", {})} - IGNORED)
    gpu = {"backlogged": dict.fromkeys(tenants, 0.0), "run": dict.fromkeys(tenants, 0.0)}
    billed = {"backlogged": dict.fromkeys(tenants, 0.0), "run": dict.fromkeys(tenants, 0.0)}
    for k, interval in samples(steps, model["offset"]):
        cost = {t: sum(coef[f] * steps[k]["work"].get(t, {}).get(f, 0) for f in coef) for t in tenants}
        total = sum(cost.values())
        prev = steps[k - 1] if k else None
        backlogged = prev is not None and all(prev["waiting"].get(t, 0) > 0 for t in tenants)
        for t in tenants:
            share = interval * cost[t] / total if total > 0 else 0.0
            service = steps[k]["accounts"].get(t, 0) - (prev["accounts"].get(t, 0) if prev else 0)
            for scope in ("run", "backlogged") if backlogged else ("run",):
                gpu[scope][t] += share
                billed[scope][t] += service

    def shares(values):
        total = sum(values.values())
        return {t: round(v / total, 4) for t, v in values.items()} if total else None

    return {"gpu_share_backlogged": shares(gpu["backlogged"]), "billed_share_backlogged": shares(billed["backlogged"]),
            "gpu_share_run": shares(gpu["run"]), "billed_share_run": shares(billed["run"]),
            "gpu_s_run": {t: round(v, 3) for t, v in gpu["run"].items()},
            "billed_run": {t: round(v, 1) for t, v in billed["run"].items()}}


def load_run(run: Path) -> tuple[dict, list[dict]]:
    meta = json.loads((run / "meta.json").read_text())
    since = meta.get("trace_start_unix")
    steps = [s for s in load_steps(run) if s.get("type") == "step" and (since is None or s["t"] >= since)]
    return meta, steps


def analyse(name: str) -> Path:
    raw = ROOT / "results/raw" / name
    runs = [load_run(p) for p in sorted(raw.iterdir()) if (p / "meta.json").exists()]
    runs = [(meta, steps) for meta, steps in runs if any("work" in s for s in steps)]
    if not runs:
        raise SystemExit(f"no runs with per-step work under {raw}")
    fits = {offset: fit([steps for _, steps in runs], offset) for offset in OFFSETS}
    model = max(fits.values(), key=lambda m: m["r2"] or -1)
    out = {"workload": runs[0][0]["workload"],
           "model": "interval_s = intercept_s + sum(coef_s[f] * f) for f in q (query tokens run as prefill), "
                    "d (decode steps), attn (causal attention pairs)",
           "r2_by_offset": {str(o): round(m["r2"], 4) for o, m in fits.items()},
           "fit": model,
           "runs": [{"arm": meta["arm"], "seed": meta["seed"], **attribute(steps, model)} for meta, steps in runs]}
    path = ROOT / "results" / name / "gpu_time.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="directories under results/raw")
    for name in parser.parse_args().names:
        print(analyse(name))


if __name__ == "__main__":
    main()
