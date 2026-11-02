"""Report math on tiny synthetic inputs."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench.report import backlogged_share, client_summary, percentile, service_timeline, step_summary, summarise_run

S = 1_000_000_000


def result(tenant, at_s, ttft_s, e2e_s, tokens=10, state="completed"):
    return {"tenant": tenant, "state": state, "scheduled_unix_ns": int(at_s * S),
            "first_output_unix_ns": int((at_s + ttft_s) * S), "last_output_unix_ns": int((at_s + e2e_s) * S),
            "finished_unix_ns": int((at_s + e2e_s) * S), "completion_tokens": tokens}


def step(t, accounts, sched_ns=1000, preempted=None, free=10, waiting=None):
    return {"type": "step", "t": t, "sched_ns": sched_ns, "scheduled": {"A": {}}, "counters": {"A": 0, "B": 0},
            "accounts": accounts, "wasted": {"A": 0, "B": 0}, "preempted": preempted or {}, "free_blocks": free,
            "waiting": waiting or {}}


class ReportTest(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile(list(range(1, 101)), 99), 99)
        self.assertEqual(percentile([5.0], 50), 5.0)
        self.assertIsNone(percentile([], 50))

    def test_client_summary_per_tenant(self):
        rows = [result("A", 0, 0.1, 1.0), result("A", 1, 0.3, 2.0),
                result("B", 0, 5.0, 9.0), result("B", 0, 0, 0, state="cancelled")]
        s = client_summary(rows)
        self.assertEqual(s["tenants"]["A"]["ttft_ms"]["p50"], 100.0)
        self.assertEqual(s["tenants"]["B"]["cancelled"], 1)
        self.assertEqual(s["tenants"]["B"]["completed"], 1)
        self.assertAlmostEqual(s["makespan_s"], 9.0)
        self.assertAlmostEqual(s["output_tokens_per_s"], 30 / 9.0)
        self.assertAlmostEqual(s["tenants"]["A"]["tpot_ms_p50"], 900 / 9)

    def test_step_summary_differences_and_preemptions(self):
        steps = [step(0.0, {"A": 0, "B": 0, "system": 0}),
                 step(1.0, {"A": 10, "B": 30, "system": 2}, sched_ns=3000, preempted={"B": 1}, free=3)]
        s = step_summary(steps)
        self.assertEqual(s["billed"], {"A": 10, "B": 30, "system": 2})
        self.assertEqual(s["preemptions"], {"B": 1})
        self.assertEqual(s["min_free_blocks"], 3)

    def test_backlogged_share_counts_only_steps_where_both_wait(self):
        both = {"A": 1, "B": 2}
        steps = [step(0, {"A": 0, "B": 0}, waiting=both), step(1, {"A": 10, "B": 30}, waiting={"B": 1}),
                 step(2, {"A": 50, "B": 40})]   # the second interval starts with A not waiting
        self.assertEqual(backlogged_share(steps, ["A", "B"]), {"A": 0.25, "B": 0.75})

    def test_service_timeline_bins_rates(self):
        steps = [step(0.0, {"A": 0}), step(0.5, {"A": 4}), step(2.5, {"A": 10})]
        self.assertEqual(service_timeline(steps, since=0.0, bin_s=2.0), {"A": [2.0, 3.0]})

    def _run_dir(self, meta):
        run = Path(tempfile.mkdtemp(prefix="fairsched-report-"))
        self.addCleanup(shutil.rmtree, run)
        (run / "meta.json").write_text(json.dumps({"arm": "vtc", "seed": 1, **meta}))
        row = {**result("A", 0, 0.1, 1.0), "finished_unix_ns": S}
        (run / "client.jsonl").write_text(json.dumps(row) + "\n")
        return run

    def test_summary_records_source_tree_not_commit(self):
        tree = {"fairsched": "a" * 40}
        s = summarise_run(self._run_dir({"source_tree": tree, "git_commit": "b" * 40}))
        self.assertEqual(s["source_tree"], tree)
        self.assertNotIn("git_commit", s)
        self.assertNotIn("source_dirty", s)
        dirty = summarise_run(self._run_dir({"source_tree": tree, "source_dirty": True}))
        self.assertTrue(dirty["source_dirty"])

    def test_summary_records_start_time_and_gpu_state(self):
        run = self._run_dir({"trace_start_unix": 1000.04})
        (run / "gpu.csv").write_text(
            "2026/11/02 10:00:00.000, 99, 19000, 2700, 10501, 61, 410.5, P2\n"
            "2026/11/02 10:00:00.500, 98, 19000, 2610, 10501, 66, 420.0, P2\n"
            "2026/11/02 10:00:01.000, 97, 19000, 2650, 10501, 64, 400.0, P2\n")
        s = summarise_run(run)
        self.assertEqual(s["started_unix"], 1000.0)
        self.assertEqual(s["gpu"]["samples"], 3)
        self.assertEqual(s["gpu"]["sm_clock_mhz"], {"p50": 2650.0, "min": 2610.0})
        self.assertEqual(s["gpu"]["temperature_c"]["max"], 66.0)
        self.assertEqual(s["gpu"]["pstates"], ["P2"])

    def test_three_column_gpu_log_has_no_gpu_summary(self):
        run = self._run_dir({})
        (run / "gpu.csv").write_text("2026/11/02 10:00:00.000, 99, 19000\n")
        self.assertNotIn("gpu", summarise_run(run))

    def test_older_run_keeps_the_source_tree_already_recorded(self):
        run = self._run_dir({"git_commit": "b" * 40})
        kept = {"fairsched": "c" * 40}
        self.assertEqual(summarise_run(run, {("vtc", 1): kept})["source_tree"], kept)
        self.assertIsNone(summarise_run(run)["source_tree"])


if __name__ == "__main__":
    unittest.main()
