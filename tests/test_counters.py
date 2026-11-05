"""Counters shared across replicas: the service, the engine-side sync and its fallback (pure Python)."""
from __future__ import annotations

import json
import socket
import time
import unittest

from fairsched.counters import CounterService, CounterStore, CounterSync, parse_publish
from fairsched.queue import TenantRequestQueue
from tests.test_policy import req, two_tenants


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StoreTest(unittest.TestCase):
    def test_remote_is_the_other_replicas_latest_totals(self):
        store = CounterStore()
        self.assertEqual(store.publish("r0", {"A": 10.0, "B": 5.0}), {"remote": {}, "replicas": 1})
        self.assertEqual(store.publish("r1", {"B": 7.0})["remote"], {"A": 10.0, "B": 5.0})
        # Cumulative totals replace the replica's previous post, so a repeat changes nothing.
        store.publish("r0", {"A": 12.0, "B": 5.0})
        store.publish("r0", {"A": 12.0, "B": 5.0})
        self.assertEqual(store.publish("r1", {"B": 9.0})["remote"], {"A": 12.0, "B": 5.0})
        self.assertEqual(store.publish("r2", {})["remote"], {"A": 12.0, "B": 14.0})

    def test_invalid_posts_are_refused(self):
        for body in ([], {"replica": "", "served": {}}, {"replica": "r", "served": []},
                     {"replica": "r", "served": {"A": -1}}, {"replica": "r", "served": {"A": float("nan")}},
                     {"replica": "r", "served": {"A": True}}, {"replica": "r", "served": {"": 1}},
                     {"replica": "r\n", "served": {}}, {"replica": "r", "served": {f"t{i}": 1 for i in range(1025)}}):
            with self.assertRaises(ValueError, msg=body):
                parse_publish(json.dumps(body).encode())
        self.assertEqual(parse_publish(b'{"replica":"r0","served":{"A":3}}'), ("r0", {"A": 3.0}))


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.service = CounterService(port=free_port()).start()
        self.addCleanup(self.service.stop)

    def client(self, replica, **kwargs):
        sync = CounterSync(self.service.url, replica, **kwargs)
        self.addCleanup(sync.stop)
        return sync

    def test_exchange_returns_each_new_remote_once(self):
        r0, r1 = self.client("r0"), self.client("r1")
        self.assertIsNone(r0.exchange({"A": 4.0}))
        self.assertTrue(r0.sync_once())
        self.assertEqual(r0.exchange({"A": 4.0}), {})
        r1.exchange({"B": 6.0})
        self.assertTrue(r1.sync_once())
        self.assertEqual(r1.exchange({"B": 6.0}), {"A": 4.0})
        self.assertTrue(r0.sync_once())
        self.assertEqual(r0.exchange({"A": 4.0}), {"B": 6.0})
        self.assertIsNone(r0.exchange({"A": 4.0}))       # nothing new since
        self.assertEqual((r0.ok, r0.failed, r0.reachable), (2, 0, True))

    def test_unreachable_service_falls_back_to_local_counters_and_recovers(self):
        port = self.service.server.server_address[1]
        state = two_tenants()
        r0, r1 = self.client("r0", timeout_s=0.5), self.client("r1")
        r1.exchange({"B": 500.0})
        r1.sync_once()
        r0.exchange(dict(state.served))
        r0.sync_once()
        state.set_remote(r0.exchange(dict(state.served)))
        self.assertEqual(state.remote, {"B": 500.0})

        self.service.stop()
        self.assertFalse(r0.sync_once())
        self.assertFalse(r0.sync_once())
        events = r0.events()
        self.assertEqual([e["state"] for e in events], ["unreachable"])   # logged once per outage
        self.assertIsNone(r0.exchange(dict(state.served)))
        # Local counting goes on; the other replicas' totals stay at their last value.
        state.served["A"] = 0.0
        self.assertEqual(state.effective_counters["B"], 500.0)

        # A restarted service is empty until each replica posts its cumulative totals again.
        self.service = CounterService(port=port).start()
        self.assertTrue(r0.sync_once())
        self.assertEqual([e["state"] for e in r0.events()], ["restored"])
        state.set_remote(r0.exchange(dict(state.served)))
        self.assertEqual(state.remote, {"B": 500.0})     # an empty reply never lowers a counter
        r1.exchange({"B": 650.0})
        r1.sync_once()
        r0.sync_once()
        state.set_remote(r0.exchange(dict(state.served)))
        self.assertEqual(state.remote, {"B": 650.0})

    def test_hung_service_times_out(self):
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        self.addCleanup(silent.close)
        sync = CounterSync(f"http://127.0.0.1:{silent.getsockname()[1]}", "r0", timeout_s=0.1)
        self.addCleanup(sync.stop)
        started = time.monotonic()
        self.assertFalse(sync.sync_once())
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(sync.events()[0]["state"], "unreachable")

    def test_thread_syncs_every_period(self):
        sync = self.client("r0", period_s=0.01).start()
        sync.exchange({"A": 1.0})
        deadline = time.monotonic() + 5
        while sync.ok < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        sync.stop()
        self.assertGreaterEqual(sync.ok, 5)
        self.assertEqual(self.service.store.snapshot()["replicas"]["r0"], {"A": 1.0})

    def test_bad_settings_are_refused(self):
        for url in ("https://127.0.0.1:1", "http://127.0.0.1:1/v1", "127.0.0.1:1"):
            with self.assertRaises(ValueError):
                CounterSync(url, "r0")
        with self.assertRaises(ValueError):
            CounterSync(self.service.url, "")
        with self.assertRaises(ValueError):
            CounterSync(self.service.url, "r0", period_s=0)


class SharedOrderingTest(unittest.TestCase):
    def test_remote_service_counts_in_selection_lift_and_victim_order(self):
        state = two_tenants()
        queue = TenantRequestQueue(state)
        b, a = req("B"), req("A")
        queue.add_request(b)
        queue.add_request(a)
        self.assertIs(queue.peek_request(), b)           # equal counters: earlier arrival
        state.set_remote({"B": 50.0})
        self.assertIs(queue.peek_request(), a)           # B was served elsewhere
        self.assertEqual(state.order_running([b, a]), [a, b])
        # A tenant joining the queue is lifted to the least-served waiting tenant's
        # counter including remote service, and only its local counter moves.
        state2 = two_tenants()
        state2.set_remote({"A": 30.0, "B": 100.0})
        q2 = TenantRequestQueue(state2)
        q2.add_request(req("B"))
        q2.add_request(req("A"))
        self.assertEqual(state2.counters["A"], 70.0)
        self.assertEqual(state2.served, {"A": 0.0, "B": 0.0})

    def test_lifts_are_not_published(self):
        state = two_tenants()
        queue = TenantRequestQueue(state)
        b = req("B")
        queue.add_request(b)
        state.begin_step(1)
        queue.pop_request()
        state.charge(b.request_id, 100, 0)
        queue.add_request(req("A"))                      # lifted to B's 100
        self.assertEqual(state.counters, {"A": 100.0, "B": 100.0})
        self.assertEqual(state.served, {"A": 0.0, "B": 100.0})

    def test_empty_remote_is_the_single_engine_policy(self):
        state = two_tenants()
        self.assertEqual(state.remote, {})
        state.set_remote({})
        self.assertEqual(state.effective_counters, state.counters)

    def simulate(self, shared: bool, sync_every: int = 1) -> float:
        """Replica 0 serves A and B, replica 1 only B; each step admits and bills one 100-token request."""
        store = CounterStore()
        states = [two_tenants(), two_tenants()]
        queues = [TenantRequestQueue(s) for s in states]
        for _ in range(400):
            queues[0].add_request(req("A"))
            queues[0].add_request(req("B"))
            queues[1].add_request(req("B"))
        admitted = {"A": 0, "B": 0}
        for step in range(300):
            if shared and step % sync_every == 0:
                for i, state in enumerate(states):
                    state.set_remote(store.publish(f"r{i}", dict(state.served))["remote"])
            for state, queue in zip(states, queues):
                state.begin_step(step)
                request = queue.pop_request()
                state.charge(request.request_id, 100, 0)
                state.settle_step()
                admitted[state.requests[request.request_id]["tenant"]] += 1
        return admitted["A"] / sum(admitted.values())

    def test_shared_counters_restore_the_global_share(self):
        self.assertAlmostEqual(self.simulate(shared=False), 0.25, delta=0.01)
        self.assertAlmostEqual(self.simulate(shared=True), 0.50, delta=0.01)
        self.assertAlmostEqual(self.simulate(shared=True, sync_every=10), 0.50, delta=0.02)


if __name__ == "__main__":
    unittest.main()
