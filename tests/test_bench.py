"""Experiment-runner configuration (pure Python; starts nothing)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench.run import CAP_INFLIGHT, FIT_W_D, IMAGE, PINS, policy_config, proxy_config
from bench.workloads import generate


class BenchConfigTest(unittest.TestCase):
    def config(self, arm):
        run = Path(tempfile.mkdtemp(prefix="fairsched-bench-"))
        self.addCleanup(shutil.rmtree, run)
        return json.loads(proxy_config(arm, run).read_text())

    def test_cap_arm_limits_every_tenant_at_the_gateway(self):
        capped, plain = self.config("fcfs-cap"), self.config("fcfs")
        for tenant in ("A", "B"):
            self.assertEqual(capped["tenants"][tenant]["max_inflight"], 16)
            self.assertEqual(plain["tenants"][tenant]["max_inflight"], 1024)
            self.assertEqual(capped["tenants"][tenant]["max_queued"], 1024)   # queued, not rejected
        self.assertFalse(capped["priority_baseline"])

    def test_cap_sweep_arms_set_their_own_cap(self):
        for arm, cap in (("fcfs-cap4", 4), ("fcfs-cap8", 8), ("fcfs-cap32", 32)):
            self.assertEqual(CAP_INFLIGHT[arm], cap)
            for tenant in ("A", "B"):
                self.assertEqual(self.config(arm)["tenants"][tenant]["max_inflight"], cap)

    def test_vtc_kv_arm_sets_the_kv_share_order(self):
        self.assertEqual(policy_config("vtc-kv", "weighted"),
                         {"tenants": {"A": {"weight": 2.0}, "B": {"weight": 1.0}}, "preempt_order": "kv_share"})
        self.assertNotIn("preempt_order", policy_config("vtc", "weighted"))

    def test_vtc_fit_arm_sets_the_fitted_decode_weight(self):
        self.assertEqual(policy_config("vtc-fit", "mixed_shapes")["w_d"], FIT_W_D)
        self.assertNotIn("w_d", policy_config("vtc", "mixed_shapes"))

    def test_mixed_shapes_gives_each_tenant_its_own_shape(self):
        rows = generate("mixed_shapes", 1)
        shapes = {(r["tenant"], len(r["body"]["prompt"]), r["body"]["max_tokens"]) for r in rows}
        self.assertEqual(shapes, {("A", 2048, 32), ("B", 128, 512)})
        self.assertEqual(len(rows), 400)

    def test_engine_image_is_pinned_by_digest(self):
        self.assertEqual(IMAGE, "vllm/vllm-openai@" + PINS["image_digest"])


if __name__ == "__main__":
    unittest.main()
