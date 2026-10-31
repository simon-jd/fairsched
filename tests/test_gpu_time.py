"""Step-time fit and GPU-time attribution on synthetic step logs (pure Python)."""
from __future__ import annotations

import random
import unittest

from bench.gpu_time import attribute, fit, least_squares

TRUE = {"c0": 0.010, "q": 2e-5, "d": 1e-4, "attn": 1e-9}


def synthetic_run(seed: int, steps: int = 400, lag: int = 1) -> list[dict]:
    """Step k's true duration shows up as the interval between records k+lag and k+lag+1."""
    rng = random.Random(seed)
    rows, accounts = [], {"A": 0.0, "B": 0.0}
    durations = []
    for _ in range(steps):
        work = {"A": {"q": rng.choice([0, 512, 2048]), "d": rng.randrange(0, 8), "attn": rng.randrange(0, 4_000_000)},
                "B": {"q": rng.choice([0, 128]), "d": rng.randrange(0, 64), "attn": rng.randrange(0, 2_000_000)}}
        for t in work:
            accounts[t] += work[t]["q"] + 2 * work[t]["d"]
        durations.append(TRUE["c0"] + sum(TRUE[f] * w[f] for w in work.values() for f in ("q", "d", "attn")))
        rows.append({"type": "step", "work": work, "waiting": {"A": 1, "B": 1}, "accounts": dict(accounts)})
    t = 0.0
    for k, row in enumerate(rows):
        if k > lag:
            t += durations[k - lag - 1]
        else:
            t += 0.001
        row["t"] = t
    return rows


class GpuTimeTest(unittest.TestCase):
    def test_least_squares_recovers_a_plane(self):
        rows = [[1.0, x, x * x % 7] for x in range(20)]
        y = [3 + 2 * r[1] - 0.5 * r[2] for r in rows]
        self.assertEqual([round(c, 6) for c in least_squares(rows, y)], [3.0, 2.0, -0.5])

    def test_fit_finds_the_logging_lag_and_the_coefficients(self):
        runs = [synthetic_run(1), synthetic_run(2)]
        fits = {offset: fit(runs, offset) for offset in (0, 1, 2, 3)}
        best = max(fits.values(), key=lambda m: m["r2"])
        self.assertEqual(best["offset"], 1)
        self.assertGreater(best["r2"], 0.999)
        self.assertAlmostEqual(best["intercept_s"], TRUE["c0"], places=6)
        for f in ("q", "d", "attn"):
            self.assertAlmostEqual(best["coef_s"][f] / TRUE[f], 1.0, places=4)

    def test_gpu_time_is_split_by_predicted_cost(self):
        run = synthetic_run(3)
        model = fit([run], 1)
        shares = attribute(run, model)
        self.assertAlmostEqual(sum(shares["gpu_share_run"].values()), 1.0, places=3)
        # Recompute A's share directly from the true model.
        expected = {"A": 0.0, "B": 0.0}
        for k in range(len(run) - 2):
            var = {t: sum(TRUE[f] * run[k]["work"][t][f] for f in ("q", "d", "attn")) for t in expected}
            interval = run[k + 2]["t"] - run[k + 1]["t"]
            for t in expected:
                expected[t] += interval * var[t] / sum(var.values())
        self.assertAlmostEqual(shares["gpu_share_run"]["A"], expected["A"] / sum(expected.values()), places=3)

    def test_negative_coefficients_are_dropped(self):
        run = synthetic_run(4)
        for row in run:                       # attn now carries no signal but noise
            for w in row["work"].values():
                w["attn"] = 0
        run[0]["work"]["A"]["attn"] = 1       # not constant, so the fit is defined
        model = fit([run], 1)
        self.assertTrue(all(c >= 0 for c in model["coef_s"].values()))


if __name__ == "__main__":
    unittest.main()
