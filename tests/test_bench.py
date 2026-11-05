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


class ReplicaBenchTest(unittest.TestCase):
    def test_session_hash_matches_the_gateway(self):
        from bench.workloads import session_hash
        # The same reference values as proxy.TestSessionHashIsFNV1a64.
        self.assertEqual(session_hash(""), 0xcbf29ce484222325)
        self.assertEqual(session_hash("a"), 0xaf63dc4c8601ec8c)

    def test_sticky_sessions_pin_A_and_spread_B(self):
        from collections import Counter
        from bench.workloads import assign_sessions, session_hash
        rows = assign_sessions(generate("aggressive", 1), 2, pinned={"A": 0})
        placed = Counter((r["tenant"], session_hash(r["session"]) % 2) for r in rows)
        self.assertEqual(placed, {("A", 0): 100, ("B", 0): 150, ("B", 1): 150})
        self.assertEqual(len({r["session"] for r in rows if r["tenant"] == "A"}), 8)

    def test_served_model_name_changes_only_the_body(self):
        default, other = generate("aggressive", 2), generate("aggressive", 2, model="m")
        self.assertEqual([r["body"]["prompt"] for r in default], [r["body"]["prompt"] for r in other])
        self.assertEqual({r["body"]["model"] for r in other}, {"m"})

    def test_global_share_counts_service_while_both_tenants_wait_anywhere(self):
        from bench.replica_report import global_share

        def step(t, waiting, a, b):
            return {"type": "step", "t": t, "waiting": waiting, "accounts": {"A": a, "B": b}}
        # Replica 0 has A and B waiting and splits evenly; replica 1 has only B.
        r0 = [step(0, {"A": 1, "B": 1}, 0, 0), step(1, {"A": 1, "B": 1}, 10, 10), step(2, {"A": 0, "B": 1}, 20, 20),
              step(3, {"A": 0, "B": 0}, 20, 40)]
        r1 = [step(0.5, {"B": 1}, 0, 0), step(1.5, {"B": 1}, 0, 20), step(2.5, {"B": 1}, 0, 40)]
        out = global_share([r0, r1])
        # Counted: r0 at t=1 and 2 (A 20, B 20), r1 at 1.5 (B 20); r0's step at 3 and
        # r1's at 2.5 come after A's queue emptied.
        self.assertEqual(out["billed"], {"A": 20.0, "B": 40.0})
        self.assertEqual(out["share"], {"A": 0.3333, "B": 0.6667})
        self.assertEqual(global_share([r0, r1], until=1.2)["billed"], {"A": 10.0, "B": 10.0})

    def test_ttft_by_upstream_splits_each_tenant_by_the_gateways_choice(self):
        from bench.replica_report import ttft_by_upstream

        def client(rid, tenant, ttft_ms, state="completed"):
            return {"rid": rid, "tenant": tenant, "state": state, "scheduled_unix_ns": 0,
                    "first_output_unix_ns": int(ttft_ms * 1e6)}
        ups = ["http://e0", "http://e1"]
        rows = [client("1", "A", 10), client("2", "B", 100), client("3", "B", 300), client("4", "B", 20),
                client("5", "B", 5, state="failed"), client("6", "B", 7)]
        terminals = [{"rid": "1", "upstream": "http://e0"}, {"rid": "2", "upstream": "http://e0"},
                     {"rid": "3", "upstream": "http://e0"}, {"rid": "4", "upstream": "http://e1"},
                     {"rid": "5", "upstream": "http://e1"}]          # rid 6 never reached an upstream
        out = ttft_by_upstream(rows, terminals, ups)
        self.assertEqual(out["A"], {"r0": {"completed": 1, "p50": 10.0, "p99": 10.0}})
        self.assertEqual(out["B"], {"r0": {"completed": 2, "p50": 100.0, "p99": 300.0},
                                    "r1": {"completed": 1, "p50": 20.0, "p99": 20.0}})

    def test_replica_arms_set_routing_mode_and_counter_url(self):
        from bench import replicas
        args = replicas.parser().parse_args(["aggressive", "gvtc-sticky", "--model", "/m", "--sync-ms", "10"])
        run = Path(tempfile.mkdtemp(prefix="fairsched-replicas-"))
        self.addCleanup(shutil.rmtree, run)
        config = json.loads(replicas.proxy_config(args, "gvtc-sticky", run, 2).read_text())
        self.assertEqual((config["routing"], len(config["upstreams"])), ("sticky", 2))
        env = replicas.engine_env(args, "gvtc-sticky", run, 1, "1")
        self.assertEqual((env["FAIRSCHED_MODE"], env["FAIRSCHED_REPLICA"], env["FAIRSCHED_SYNC_MS"],
                          env["CUDA_VISIBLE_DEVICES"]), ("vtc", "r1", "10.0", "1"))
        self.assertEqual(env["FAIRSCHED_COUNTER_URL"], "http://127.0.0.1:8090")
        local = replicas.engine_env(args, "vtc-rr", run, 0, "0")
        self.assertNotIn("FAIRSCHED_COUNTER_URL", local)
        self.assertEqual(replicas.engine_env(args, "fcfs-rr", run, 0, "0")["FAIRSCHED_MODE"], "passthrough")
        rr = json.loads(replicas.proxy_config(args, "vtc-rr", run, 2).read_text())
        self.assertEqual(rr["routing"], "round_robin")
