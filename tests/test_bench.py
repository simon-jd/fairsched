"""Experiment-runner configuration (pure Python; starts nothing)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from bench.run import CAP_INFLIGHT, IMAGE, PINS, proxy_config


class BenchConfigTest(unittest.TestCase):
    def config(self, arm):
        run = Path(tempfile.mkdtemp(prefix="fairsched-bench-"))
        self.addCleanup(shutil.rmtree, run)
        return json.loads(proxy_config(arm, run).read_text())

    def test_cap_arm_limits_every_tenant_at_the_gateway(self):
        capped, plain = self.config("fcfs-cap"), self.config("fcfs")
        for tenant in ("A", "B"):
            self.assertEqual(capped["tenants"][tenant]["max_inflight"], CAP_INFLIGHT)
            self.assertEqual(plain["tenants"][tenant]["max_inflight"], 1024)
            self.assertEqual(capped["tenants"][tenant]["max_queued"], 1024)   # queued, not rejected
        self.assertFalse(capped["priority_baseline"])

    def test_engine_image_is_pinned_by_digest(self):
        self.assertEqual(IMAGE, "vllm/vllm-openai@" + PINS["image_digest"])


if __name__ == "__main__":
    unittest.main()
