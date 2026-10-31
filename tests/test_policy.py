"""Policy and queue behaviour, pure Python (no vLLM needed)."""
from __future__ import annotations

from types import SimpleNamespace
import random
import unittest

from fairsched.policy import DEFAULT, SYSTEM, PolicyConfig, PolicyState, TenantPolicy
from fairsched.intervals import merge, overlap
from fairsched.queue import TenantRequestQueue

_ids = iter(range(10**9))


def req(tenant=None, prompt=100, computed=0, arrival=None):
    n = next(_ids)
    extra = {} if tenant is None else {"fairsched_tenant": tenant}
    return SimpleNamespace(request_id=f"r{n}", arrival_time=float(n if arrival is None else arrival),
                           num_prompt_tokens=prompt, num_computed_tokens=computed,
                           sampling_params=SimpleNamespace(extra_args=extra))


def two_tenants(**kwargs):
    weights = kwargs.pop("weights", (1.0, 1.0))
    return PolicyState(PolicyConfig(tenants={"A": TenantPolicy(weights[0]), "B": TenantPolicy(weights[1])},
                                    **kwargs))


class PolicyTest(unittest.TestCase):
    def test_lift_on_rebacklog(self):
        state = two_tenants()
        queue = TenantRequestQueue(state)
        b1, b2 = req("B"), req("B")
        queue.add_request(b1)
        queue.add_request(b2)
        state.begin_step(1)
        queue.pop_request()
        state.charge(b1.request_id, 100, 0)
        state.counters["B"] += 400                 # B has been served while still waiting
        queue.add_request(req("A"))
        self.assertEqual(state.counters["A"], 500.0)

    def test_lift_uses_last_departure_when_nobody_is_waiting(self):
        state = two_tenants()
        queue = TenantRequestQueue(state)
        a = req("A", prompt=100)
        queue.add_request(a)
        state.begin_step(1)
        queue.pop_request()                        # A's queue empties: last_left = 0 + 100 reserved
        state.charge(a.request_id, 100, 0)
        state.charge(a.request_id, 0, 100)         # A keeps decoding: counter 300
        queue.add_request(req("B"))
        self.assertEqual(state.counters["B"], 100.0)

    def test_tenant_with_only_running_work_is_lifted_on_its_next_arrival(self):
        # A keeps one long request decoding while B stays queued and is served
        # heavily. A's next burst must start at B's level, not from its stale
        # counter, or it would monopolise the engine until the gap closed.
        state = two_tenants()
        queue = TenantRequestQueue(state)
        long_a = req("A", prompt=10)
        queue.add_request(long_a)
        state.begin_step(1)
        queue.pop_request()
        for _ in range(3):
            queue.add_request(req("B"))
        state.counters["B"] = 10_000.0
        state.charge(long_a.request_id, 0, 1)
        queue.add_request(req("A"))
        self.assertEqual(state.counters["A"], 10_000.0)

    def test_weighted_share_two_backlogged(self):
        state = two_tenants(weights=(2.0, 1.0))
        queue = TenantRequestQueue(state)
        for _ in range(400):
            queue.add_request(req("A", prompt=10))
            queue.add_request(req("B", prompt=10))
        served = {"A": 0, "B": 0}
        for step in range(1, 201):
            state.begin_step(step)
            r = queue.pop_request()
            tenant = state.requests[r.request_id]["tenant"]
            state.charge(r.request_id, 10, 0)
            served[tenant] += 10
            state.settle_step()
        self.assertAlmostEqual(served["A"] / served["B"], 2.0, delta=0.1)

    def test_recompute_charged_to_system_by_default(self):
        state = two_tenants()
        a = req("A")
        state.register(a)
        state.begin_step(1)
        # 100 recomputed prompt positions, 4 recomputed output positions, 6 new decode tokens.
        state.charge(a.request_id, 100, 10, recompute_prefill=100, recompute_decode=4)
        self.assertEqual(state.accounts[SYSTEM], 104.0)  # recompute runs as prefill: w_p for all of it
        self.assertEqual(state.accounts["A"], 12.0)      # 6 new decode tokens at w_d=2
        self.assertEqual(state.counters["A"], 12.0)

    def test_recompute_can_be_billed_to_tenant(self):
        state = two_tenants(charge_recompute="tenant")
        a = req("A")
        state.register(a)
        state.begin_step(1)
        state.charge(a.request_id, 100, 0, recompute_prefill=100)
        self.assertEqual(state.counters["A"], 100.0)
        self.assertEqual(state.accounts[SYSTEM], 0.0)

    def test_wasted_marked_once(self):
        state = two_tenants(charge_wasted=SYSTEM)
        a = req("A")
        state.register(a)
        state.begin_step(7)
        state.charge(a.request_id, 0, 1)
        counter = state.counters["A"]
        self.assertEqual(state.mark_wasted(7, a.request_id), 2.0)
        self.assertEqual(state.mark_wasted(7, a.request_id), 0.0)
        self.assertEqual(state.accounts, {"A": 0.0, "B": 0.0, SYSTEM: 2.0})
        self.assertEqual(state.counters["A"], counter)     # history is not rewritten

    def test_unknown_tenants_share_the_default_tenant(self):
        state = two_tenants()
        z, y, bad = req("Z", prompt=10), req("Y", prompt=10), req(["not", "a", "name"], prompt=10)
        for r in (z, y, bad):
            self.assertEqual(state.register(r)["tenant"], DEFAULT)
        self.assertEqual(state.requests[z.request_id]["claimed_tenant"], "Z")
        self.assertEqual(state.unknown_requests, 3)
        # Billed and ordered like any tenant, at weight 1, but never as "Z" or "Y".
        state.charge(z.request_id, 10, 0)
        state.charge(y.request_id, 10, 0)
        self.assertEqual(state.accounts[DEFAULT], 20.0)
        self.assertEqual(state.counters[DEFAULT], 20.0)
        self.assertNotIn("Z", state.counters)
        queue = TenantRequestQueue(state)
        a = req("A", prompt=10)
        queue.add_request(a)
        queue.add_request(req("Z", prompt=10))
        self.assertIs(queue.pop_request(), a)       # A has been served less than default

    def test_missing_tenant_defaults(self):
        state = PolicyState()
        meta = state.register(req(None))
        self.assertEqual(meta["tenant"], "default")
        self.assertNotIn("claimed_tenant", meta)
        self.assertEqual(state.unknown_requests, 0)

    def test_order_running_puts_most_overserved_last(self):
        state = two_tenants()
        a1, a2, b1 = req("A"), req("A"), req("B")
        for r in (a1, a2, b1):
            state.register(r)
        state.begin_step(1)
        state.charge(a1.request_id, 1000, 0)
        self.assertEqual(state.order_running([a2, a1, b1]), [b1, a1, a2])

    def test_kv_share_order_protects_a_tenant_within_its_kv_share(self):
        def running(order):
            state = two_tenants(preempt_order=order)
            b_old, b_new = req("B", computed=900), req("B", computed=800)
            a_new = req("A", computed=40)
            for r in (b_old, b_new, a_new):
                state.register(r)
            state.counters.update(A=50.0, B=40.0)   # A just lifted above B, as after a lift
            return state.order_running([a_new, b_new, b_old]), (b_old, b_new, a_new)
        ordered, (b_old, b_new, a_new) = running("counter")
        self.assertIs(ordered[-1], a_new)             # the light tenant is the victim
        ordered, (b_old, b_new, a_new) = running("kv_share")
        self.assertEqual(ordered, [a_new, b_old, b_new])   # B holds 98% of the KV: its newest goes

    def test_kv_share_order_uses_weights_and_keeps_counter_order(self):
        state = two_tenants(weights=(3.0, 1.0), preempt_order="kv_share")
        a1, a2, b1 = req("A", computed=600), req("A", computed=600), req("B", computed=400)
        for r in (a1, a2, b1):
            state.register(r)
        state.counters.update(A=10.0, B=20.0)
        # A holds 75% with a 75% share, B 25% with 25%: nobody is over, so counter order.
        self.assertEqual(state.order_running([b1, a2, a1]), [a1, a2, b1])
        with self.assertRaises(ValueError):
            PolicyConfig(preempt_order="largest")

    def test_same_step_admissions_see_reserved_prefill(self):
        state = two_tenants()
        queue = TenantRequestQueue(state)
        for _ in range(3):
            queue.add_request(req("A", prompt=1000))
        b = req("B", prompt=10, arrival=10**6)
        queue.add_request(b)
        state.begin_step(1)
        first = queue.pop_request()
        self.assertEqual(state.requests[first.request_id]["tenant"], "A")
        self.assertIs(queue.pop_request(), b)    # A's reserved 1000-token prefill now counts

    def test_queue_matches_fcfs_prepend_and_remove_semantics(self):
        state = PolicyState()
        queue = TenantRequestQueue(state)
        rs = [req(None) for _ in range(4)]
        queue.add_request(rs[0])
        queue.prepend_requests(rs[1:3])
        self.assertEqual([r.request_id for r in queue], [rs[2].request_id, rs[1].request_id, rs[0].request_id])
        queue.remove_requests([rs[1], rs[3]])
        self.assertIs(queue.peek_request(), rs[2])
        self.assertIs(queue.pop_request(), rs[2])
        queue.remove_request(rs[0])
        self.assertFalse(queue)
        self.assertEqual(len(queue), 0)
        with self.assertRaises(IndexError):
            queue.peek_request()


class IntervalTest(unittest.TestCase):
    def test_merge_matches_sort_and_coalesce(self):
        def reference(intervals, start, end):
            out = []
            for a, b in sorted([*intervals, (start, end)]):
                if out and a <= out[-1][1]:
                    out[-1] = (out[-1][0], max(b, out[-1][1]))
                else:
                    out.append((a, b))
            return out
        rng = random.Random(0)
        for _ in range(5000):
            intervals = []
            for _ in range(rng.randrange(6)):
                s = rng.randrange(50)
                intervals = reference(intervals, s, s + rng.randrange(1, 10))
            s = rng.randrange(60)
            e = s + rng.randrange(1, 12)
            self.assertEqual(merge(list(intervals), s, e), reference(intervals, s, e))

    def test_overlap_counts_reexecuted_positions(self):
        self.assertEqual(overlap([(0, 10), (20, 30)], 5, 25), 10)
        self.assertEqual(overlap([], 0, 100), 0)


if __name__ == "__main__":
    unittest.main()
