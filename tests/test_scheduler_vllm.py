"""The plugin inside the real vLLM 0.31.0 scheduling loop (CPU only).

Runs in the pinned container: ``scripts/test.sh``. Skipped where vLLM is absent.
"""
from __future__ import annotations

import importlib.util
import unittest

HAVE_VLLM = importlib.util.find_spec("vllm") is not None
if HAVE_VLLM:
    from tests.vllm_harness import Engine, make_request, make_scheduler


@unittest.skipUnless(HAVE_VLLM, "needs the vllm/vllm-openai:v0.31.0 container")
class SchedulerInVllmTest(unittest.TestCase):
    def noisy_neighbor(self, mode):
        """B queues 8 x 2048-token prompts; A's short request arrives right after."""
        engine = Engine(make_scheduler(mode=mode, budget=512))
        for _ in range(8):
            engine.add(make_request("B", 2048, max_tokens=16))
        engine.run_step()
        engine.run_step()
        a = make_request("A", 64, max_tokens=4)
        engine.add(a)
        arrived = engine.step
        engine.run_until_idle()
        return engine, engine.first_token[a.request_id] - arrived

    def test_underserved_tenant_is_admitted_ahead_of_the_backlog(self):
        _, vtc_wait = self.noisy_neighbor("vtc")
        _, fcfs_wait = self.noisy_neighbor("passthrough")
        # FCFS serves the other seven queued 2048-token B prompts first (~28 steps at 512/step).
        self.assertGreaterEqual(fcfs_wait, 25)
        # VTC only waits for the B prefill already running when A arrived.
        self.assertLessEqual(vtc_wait, 6)

    PROMPT, OUTPUT = 32, 200

    def kv_pressure(self, mode):
        """Three B decodes and one later A decode share 40 KV blocks, which run out mid-decode.

        A's counter is lifted to B's on arrival; B then accrues 3x faster, so by
        the time KV runs out (~100 steps later) B is the most over-served tenant.
        """
        engine = Engine(make_scheduler(mode=mode, budget=4096, num_blocks=40))
        for _ in range(3):
            engine.add(make_request("B", self.PROMPT, max_tokens=self.OUTPUT))
        for _ in range(4):
            engine.run_step()
        engine.add(make_request("A", self.PROMPT, max_tokens=self.OUTPUT))
        engine.run_until_idle()
        return engine

    def test_kv_pressure_preempts_the_most_overserved_tenant(self):
        vtc = self.kv_pressure("vtc")
        fcfs = self.kv_pressure("passthrough")
        self.assertTrue(vtc.preempted, "scenario must create KV pressure")
        self.assertEqual(set(vtc.preempted), {"B"})
        self.assertIn("A", fcfs.preempted)        # native FCFS evicts the newest request
        self.assertEqual(len(vtc.finished), 4)

    def test_recompute_after_preemption_is_billed_to_system(self):
        engine = self.kv_pressure("vtc")
        accounts = engine.s.fairsched.accounts
        self.assertGreater(accounts["system"], 0)
        # Tenants pay for each position once: the prompt plus OUTPUT - 1 decode
        # positions (the last sampled token is never fed back). Re-executing
        # positions lost to preemption goes to "system". Async scheduling may add
        # one over-scheduled decode token per request (billed, then marked wasted).
        tenant_billed = accounts["A"] + accounts["B"]
        self.assertAlmostEqual(tenant_billed, 4 * (self.PROMPT + 2 * (self.OUTPUT - 1)), delta=4 * 2.0)

    def test_long_prompts_finish_under_kv_pressure(self):
        # The workload that livelocked the old quantum-preemption policy
        # (docs/livelock.md). vLLM admits the long prompts nearly one at a time
        # (running work spends the 512-token budget first), so the KV cache has
        # to be close to one long request's size before running requests
        # contend: 450 blocks = 7,200 tokens, one prompt needs 6,176. Native
        # preemption must fire, take the over-served tenant, and every request
        # must still finish.
        engine = Engine(make_scheduler(mode="vtc", budget=512, num_blocks=450))
        longs = [make_request("B", 6144, max_tokens=32) for _ in range(3)]
        for r in longs:
            engine.add(r)
        shorts = [make_request("A", 128, max_tokens=32) for _ in range(10)]
        for r in shorts:
            engine.add(r)
        steps = engine.run_until_idle(limit=5000)
        self.assertTrue(engine.preempted, "scenario must create KV pressure")
        self.assertEqual(set(engine.preempted), {"B"})
        self.assertTrue(all(r.request_id in engine.finished for r in longs + shorts))
        # No request is preempted over and over (the livelock signature) ...
        self.assertLessEqual(max(r.num_preemptions for r in longs + shorts), 2)
        # ... and total work stays within a small factor of the useful work.
        work = 3 * 6144 + 10 * 128
        self.assertLess(steps, 3 * work / 512 + 3 * 32)

    def test_abort_with_step_in_flight_marks_its_tokens_wasted(self):
        s = make_scheduler(mode="vtc")
        engine = Engine(s)
        a = make_request("A", 32, max_tokens=50)
        engine.add(a)
        for _ in range(4):
            engine.run_step()          # prefill done, decoding; one step still in flight
        self.assertEqual(len(engine.inflight), 1)
        engine.abort(a.request_id)
        engine.drain()
        self.assertEqual(s.fairsched.wasted["A"], 2.0)   # one in-flight decode token, w_d = 2
        self.assertEqual(s.fairsched.requests, {})        # metadata released once nothing refers to it

    def test_unknown_tenant_is_served_as_default(self):
        for mode in ("vtc", "passthrough"):
            s = make_scheduler(mode=mode)           # the policy lists only A and B
            engine = Engine(s)
            engine.add(make_request("Z", 32, max_tokens=4))
            engine.add(make_request("A", 32, max_tokens=4))
            engine.run_until_idle()
            self.assertEqual(len(engine.finished), 2)
            self.assertGreater(s.fairsched.accounts["default"], 0)
            self.assertEqual(s.fairsched.unknown_requests, 1)

    def test_prefix_cache_hits_are_not_billed(self):
        s = make_scheduler(mode="vtc", budget=4096)
        engine = Engine(s)
        shared = list(range(1000, 1000 + 256))       # 16 full blocks
        engine.add(make_request("A", 0, max_tokens=1, tokens=shared + [5, 6, 7]))
        engine.run_until_idle()
        first = s.fairsched.accounts["A"]
        engine.add(make_request("A", 0, max_tokens=1, tokens=shared + [8, 9, 10]))
        engine.run_until_idle()
        self.assertLess(s.fairsched.accounts["A"] - first, 0.25 * first)

    def test_weights_split_service_when_both_tenants_are_backlogged(self):
        s = make_scheduler(mode="vtc", weights={"A": 2.0, "B": 1.0}, budget=256, max_seqs=4)
        engine = Engine(s)
        for _ in range(40):
            engine.add(make_request("A", 64, max_tokens=64))
            engine.add(make_request("B", 64, max_tokens=64))
        for _ in range(300):
            engine.run_step()
        a, b = s.fairsched.accounts["A"], s.fairsched.accounts["B"]
        self.assertAlmostEqual(a / b, 2.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()
