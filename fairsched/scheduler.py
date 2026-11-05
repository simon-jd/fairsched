"""VTC fair scheduling as an out-of-tree vLLM V1 scheduler (``--scheduler-cls``).

Three hooks into the unmodified vLLM 0.31.0 scheduler do all the work:

* ``self.waiting`` is replaced by :class:`TenantRequestQueue`, so admission takes
  the oldest request of the least-served tenant.
* Before the native ``schedule()`` runs, ``self.running`` is sorted least-served
  first. vLLM spends the token budget in that order and preempts ``running[-1]``
  under KV pressure, so the most over-served tenant yields both budget and KV.
* ``_update_after_schedule`` bills the tokens vLLM actually scheduled, before it
  advances ``num_computed_tokens``. Prefix-cache hits were already skipped, so
  they are never billed. Positions executed before a preemption are recompute.

The policy never preempts a request itself (see docs/livelock.md for why).
"""
from __future__ import annotations

import os
import time
from importlib.metadata import version
from pathlib import Path

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from .audit import StepLog
from .counters import CounterSync
from .policy import SYSTEM, PolicyConfig, PolicyState
from .intervals import merge as _merge, overlap as _overlap
from .queue import TenantRequestQueue

PINNED_VLLM = "0.31.0"


class TenantSchedulerMixin:
    """Shared by the async (default) and synchronous scheduler variants.

    ``FAIRSCHED_MODE=vtc`` enables fair ordering. ``FAIRSCHED_MODE=passthrough``
    keeps the native policy (FCFS or priority) but records the same per-tenant
    accounting, so baselines and VTC are measured identically.

    ``FAIRSCHED_COUNTER_URL`` (VTC only) shares counters with other replicas
    through ``fairsched.counters``; ``FAIRSCHED_REPLICA`` names this engine and
    ``FAIRSCHED_SYNC_MS`` sets the sync period (default 100 ms).
    """

    def __init__(self, *args, **kwargs):
        if version("vllm") != PINNED_VLLM:
            raise RuntimeError(f"fairsched hooks vLLM internals and is pinned to {PINNED_VLLM}")
        super().__init__(*args, **kwargs)
        self.fairsched_mode = os.environ.get("FAIRSCHED_MODE", "vtc")
        if self.fairsched_mode not in {"vtc", "passthrough"}:
            raise ValueError("FAIRSCHED_MODE must be 'vtc' or 'passthrough'")
        if self.fairsched_mode == "vtc" and self.scheduler_config.policy != "fcfs":
            raise ValueError("VTC replaces FCFS ordering; run it with --scheduling-policy fcfs")
        if self.connector is not None or self.num_spec_tokens or self.lora_config:
            raise ValueError("fairsched supports text-only serving without KV connector, spec decode or LoRA")
        parallel = self.parallel_config
        if (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size) != (1, 1, 1):
            raise ValueError("fairsched keeps per-engine counters and supports a single GPU (TP=PP=DP=1)")
        path = os.environ.get("FAIRSCHED_POLICY")
        self.fairsched = PolicyState(PolicyConfig.load(path) if path else PolicyConfig())
        if self.fairsched_mode == "vtc":
            native, self.waiting = self.waiting, TenantRequestQueue(self.fairsched)
            for request in native:
                self.waiting.add_request(request)
        # Counters shared with other replicas (off unless FAIRSCHED_COUNTER_URL is set).
        self.fairsched_sync: CounterSync | None = None
        self.fairsched_sync_ns: int | None = None
        sync_url = os.environ.get("FAIRSCHED_COUNTER_URL")
        sync = None
        if sync_url:
            if self.fairsched_mode != "vtc":
                raise ValueError("FAIRSCHED_COUNTER_URL needs FAIRSCHED_MODE=vtc")
            sync = {"url": sync_url, "replica": os.environ.get("FAIRSCHED_REPLICA") or f"pid-{os.getpid()}",
                    "period_ms": float(os.environ.get("FAIRSCHED_SYNC_MS", "100"))}
            self.fairsched_sync = CounterSync(sync["url"], sync["replica"], period_s=sync["period_ms"] / 1000)
        output = Path(os.environ.get("FAIRSCHED_OUTPUT", "fairsched-output"))
        self.fairsched_log = StepLog(output / f"steps-{os.getpid()}.jsonl")
        self.fairsched_log.write({"type": "start", "mode": self.fairsched_mode,
                                  "policy": self.fairsched.config.to_dict(),
                                  "native_policy": self.scheduler_config.policy,
                                  "token_budget": self.max_num_scheduled_tokens,
                                  "num_blocks": len(self.kv_cache_manager.block_pool.blocks),
                                  "block_size": self.block_size, "async": isinstance(self, AsyncScheduler),
                                  **({"counter_sync": sync} if sync else {})})
        if self.fairsched_sync is not None:
            self.fairsched_sync.start()
        self.fairsched_executed: dict[str, list[tuple[int, int]]] = {}   # request -> executed positions
        self.fairsched_pending: dict[int, int] = {}                      # id(SchedulerOutput) -> step
        self.fairsched_preempted: dict[str, int] = {}
        self.fairsched_finished: set[str] = set()                        # freed, metadata not yet dropped
        self.fairsched_work: dict[str, dict[str, int]] = {}              # this step's GPU work, per tenant

    # Admission ---------------------------------------------------------------

    def add_request(self, request):
        self.fairsched.register(request)   # unlisted tenants are served as "default"
        super().add_request(request)

    # Scheduling --------------------------------------------------------------

    def schedule(self, throttle_prefills: bool = False):
        step = self.current_step + 1
        self.fairsched.begin_step(step)
        self.fairsched_preempted = {}
        self.fairsched_work = {}
        started = time.perf_counter_ns()
        sync_ns = None
        if self.fairsched_sync is not None:
            # Hand the sync thread this engine's totals and take the latest
            # remote ones; no network I/O on this thread.
            remote = self.fairsched_sync.exchange(dict(self.fairsched.served))
            if remote is not None:
                self.fairsched.set_remote(remote)
            sync_ns = time.perf_counter_ns() - started
        if self.fairsched_mode == "vtc":
            self.running[:] = self.fairsched.order_running(self.running)
        output = super().schedule(throttle_prefills=throttle_prefills)
        elapsed = time.perf_counter_ns() - started
        self.fairsched_sync_ns = sync_ns
        self.fairsched.settle_step()
        if output.num_scheduled_tokens:
            # Keyed by object identity: the engine holds this output until it
            # calls update_from_output with it. An entry left by an output that
            # was never reported back cannot claim this one's step.
            stale = self.fairsched_pending.pop(id(output), None)
            if stale is not None:
                self.fairsched.release_step(stale)
            self.fairsched_pending[id(output)] = step
        else:
            self.fairsched.release_step(step)   # nothing billed; the engine may never report it
        if self.fairsched_sync is not None:
            for event in self.fairsched_sync.events():
                self.fairsched_log.write({**event, "step": step})
        if output.num_scheduled_tokens or self.fairsched_preempted:
            self._fairsched_log_step(step, elapsed)
        return output

    def _update_after_schedule(self, scheduler_output):
        # Runs before the native method advances num_computed_tokens, so
        # [num_computed_tokens, +scheduled) are exactly the positions executed now.
        for request_id, tokens in scheduler_output.num_scheduled_tokens.items():
            request = self.requests[request_id]
            start, prompt = request.num_computed_tokens, request.num_prompt_tokens
            end = start + tokens
            prefill = max(0, min(end, prompt) - start)
            done = self.fairsched_executed.get(request_id, [])
            self.fairsched.charge(request_id, prefill, tokens - prefill,
                                  recompute_prefill=_overlap(done, start, min(end, prompt)),
                                  recompute_decode=_overlap(done, max(start, prompt), end))
            self.fairsched_executed[request_id] = _merge(done, start, end)
            # What the GPU does for it, for fitting step time: decode steps
            # (one new position after the prompt), query tokens run as prefill
            # (prompt chunks and recompute), and causal attention pairs.
            work = self.fairsched_work.setdefault(self.fairsched.requests[request_id]["tenant"],
                                                  {"q": 0, "d": 0, "attn": 0})
            if tokens == 1 and start >= prompt:
                work["d"] += 1
            else:
                work["q"] += tokens
            work["attn"] += tokens * start + tokens * (tokens + 1) // 2
        super()._update_after_schedule(scheduler_output)

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        tenant = self.fairsched.requests[request.request_id]["tenant"]
        self.fairsched_preempted[tenant] = self.fairsched_preempted.get(tenant, 0) + 1
        return super()._preempt_request(request, timestamp, drop_stale_output)

    # Completion --------------------------------------------------------------

    def update_from_output(self, scheduler_output, model_runner_output):
        step = self.fairsched_pending.pop(id(scheduler_output), None)
        if step is not None:
            # With async scheduling a request can be aborted while a step that
            # includes it is still on the GPU; that step's tokens are wasted.
            for record in self.fairsched.ledger.get(step, []):
                request = self.requests.get(record["request_id"])
                if request is None or request.is_finished():
                    self.fairsched.mark_wasted(step, record["request_id"])
        result = super().update_from_output(scheduler_output, model_runner_output)
        if step is not None:
            self.fairsched.release_step(step)
            if self.fairsched_finished:
                # Drop metadata of finished requests no in-flight step still bills.
                in_flight = {r["request_id"] for s in self.fairsched_pending.values()
                             for r in self.fairsched.ledger.get(s, [])}
                for request_id in self.fairsched_finished - in_flight:
                    self.fairsched.forget(request_id)
                    self.fairsched_executed.pop(request_id, None)
                self.fairsched_finished &= in_flight
        return result

    def _free_request(self, request, delay_free_blocks=False):
        self.fairsched.finish(request.request_id)
        self.fairsched_finished.add(request.request_id)
        return super()._free_request(request, delay_free_blocks=delay_free_blocks)

    def shutdown(self):
        try:
            return super().shutdown()
        finally:
            if self.fairsched_sync is not None:
                self.fairsched_sync.stop()
                for event in self.fairsched_sync.events():
                    self.fairsched_log.write(event)
                rtt = sorted(self.fairsched_sync.rtt_us)
                self.fairsched_log.write({"type": "counter_sync_end", "ok": self.fairsched_sync.ok,
                                          "failed": self.fairsched_sync.failed,
                                          "rtt_us": {"p50": rtt[len(rtt) // 2], "p99": rtt[int(len(rtt) * 0.99)]}
                                          if rtt else None})
            self.fairsched_log.close()

    # Logging -----------------------------------------------------------------

    def _fairsched_log_step(self, step: int, elapsed_ns: int) -> None:
        per_tenant: dict[str, dict[str, float]] = {}
        for record in self.fairsched.ledger.get(step, []):
            row = per_tenant.setdefault(record["tenant"], {"prefill": 0, "decode": 0, "recompute": 0})
            for key in ("prefill", "decode", "recompute"):
                row[key] += record[key]
        running: dict[str, int] = {}
        for request in self.running:
            tenant = self.fairsched.requests[request.request_id]["tenant"]
            running[tenant] = running.get(tenant, 0) + 1
        waiting = self.waiting.counts() if self.fairsched_mode == "vtc" else self._fairsched_count(self.waiting)
        self.fairsched_log.write({
            "type": "step", "step": step, "t": time.time(), "sched_ns": elapsed_ns,
            "scheduled": per_tenant, "work": self.fairsched_work, "running": running, "waiting": waiting,
            "preempted": self.fairsched_preempted, "counters": dict(self.fairsched.counters),
            "accounts": dict(self.fairsched.accounts), "wasted": dict(self.fairsched.wasted),
            "free_blocks": self.kv_cache_manager.block_pool.get_num_free_blocks(),
            **({"unknown_requests": self.fairsched.unknown_requests} if self.fairsched.unknown_requests else {}),
            **(self._fairsched_sync_record() if self.fairsched_sync is not None else {})})

    def _fairsched_sync_record(self) -> dict:
        sync = self.fairsched_sync
        return {"remote": dict(self.fairsched.remote), "sync_ns": self.fairsched_sync_ns,
                "sync": {"ok": sync.ok, "failed": sync.failed, "reachable": sync.reachable,
                         "rtt_us": round(sync.rtt_us[-1], 1) if sync.rtt_us else None}}

    def _fairsched_count(self, requests) -> dict[str, int]:
        counts: dict[str, int] = {}
        for request in requests:
            meta = self.fairsched.requests.get(request.request_id)
            tenant = meta["tenant"] if meta else SYSTEM
            counts[tenant] = counts.get(tenant, 0) + 1
        return counts


class TenantAsyncScheduler(TenantSchedulerMixin, AsyncScheduler):
    """Default: vLLM's async scheduler, which overlaps scheduling with the GPU step."""


class TenantSyncScheduler(TenantSchedulerMixin, Scheduler):
    """Synchronous control, for runs that need ``--no-async-scheduling``."""
