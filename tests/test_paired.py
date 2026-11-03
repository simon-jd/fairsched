"""Paired same-seed comparison on hand-made summaries."""
from __future__ import annotations

import statistics
import unittest

from bench.paired import analyse_runs, mean_interval


def run(arm, seed, tps, started, clock=2600.0):
    return {"arm": arm, "seed": seed, "started_unix": started, "client": {"output_tokens_per_s": tps},
            "gpu": {"sm_clock_mhz": {"p50": clock}, "temperature_c": {"max": 80.0}}}


class PairedTest(unittest.TestCase):
    def test_mean_interval_matches_t_formula(self):
        values = [1.0, 2.0, 4.0]
        m, lo, hi = mean_interval(values)
        half = 4.303 * statistics.stdev(values) / 3 ** 0.5
        self.assertAlmostEqual(m, 7 / 3)
        self.assertAlmostEqual(lo, 7 / 3 - half)
        self.assertAlmostEqual(hi, 7 / 3 + half)
        with self.assertRaises(ValueError):
            mean_interval([1.0])

    def test_consistent_gap_is_resolved_and_order_recorded(self):
        # b is 1% faster than a in every seed; a runs first in seeds 1 and 3.
        runs = [run("a", 1, 100.0, 10), run("b", 1, 101.0, 20),
                run("b", 2, 202.0, 10), run("a", 2, 200.0, 20),
                run("a", 3, 150.0, 10), run("b", 3, 151.5, 20)]
        out = analyse_runs(runs, [("b", "a")])
        pair = out["pairs"][0]
        self.assertEqual(out["seeds"], [1, 2, 3])
        self.assertEqual(pair["diff_pct"], [1.0, 1.0, 1.0])
        self.assertEqual(pair["ci95_pct"], [1.0, 1.0])
        self.assertTrue(pair["resolved"])
        self.assertEqual(pair["a_ran_first"], [False, True, False])
        self.assertEqual(out["arms"]["a"]["position"], [1, 2, 1])
        self.assertEqual(out["arms"]["a"]["median"], 150.0)
        self.assertAlmostEqual(out["arms"]["a"]["spread_pct"], 100 * 100 / 150, places=3)

    def test_gap_of_either_sign_is_unresolved(self):
        runs = [run("a", 1, 100.0, 1), run("b", 1, 101.0, 2),
                run("a", 2, 100.0, 1), run("b", 2, 99.0, 2),
                run("a", 3, 100.0, 1), run("b", 3, 100.5, 2)]
        pair = analyse_runs(runs, [("b", "a")])["pairs"][0]
        self.assertLess(pair["ci95_pct"][0], 0)
        self.assertGreater(pair["ci95_pct"][1], 0)
        self.assertFalse(pair["resolved"])

    def test_position_effect(self):
        # The second run of every seed is 2% slower than the first, whatever the arm.
        runs = [run("a", 1, 102.0, 1), run("b", 1, 100.0, 2),
                run("b", 2, 102.0, 1), run("a", 2, 100.0, 2)]
        effect = analyse_runs(runs, [])["position_effect_pct"]
        self.assertAlmostEqual(effect["1"], 100 * (102 / 101 - 1), places=3)
        self.assertAlmostEqual(effect["2"], 100 * (100 / 101 - 1), places=3)

    def test_missing_run_or_arm_is_an_error(self):
        runs = [run("a", 1, 100.0, 1), run("b", 1, 101.0, 2), run("a", 2, 100.0, 1)]
        with self.assertRaises(ValueError):
            analyse_runs(runs, [("b", "a")])
        with self.assertRaises(ValueError):
            analyse_runs(runs[:2], [("c", "a")])


if __name__ == "__main__":
    unittest.main()
