"""Drive the real vLLM 0.31.0 scheduler on CPU, without a model or GPU.

The native ``Scheduler``/``AsyncScheduler``, KV cache manager, block pool and
prefix cache all run for real; only the model forward pass is replaced by a
fake runner output that samples token 7 for every request that finished its
prefill this step. Run inside the ``vllm/vllm-openai:v0.31.0`` container.
"""
from __future__ import annotations

from collections import deque
import itertools
import json
import os
from pathlib import Path
import random
import tempfile

import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from fairsched.scheduler import TenantAsyncScheduler, TenantSyncScheduler

MODEL = os.environ.get("FAIRSCHED_TEST_MODEL", "/models/Qwen2.5-1.5B-Instruct")
BLOCK = 16
init_none_hash(sha256)
_HASHER = get_request_block_hasher(BLOCK, sha256)
_ids = itertools.count()


def make_scheduler(*, mode="vtc", weights=None, budget=512, num_blocks=2000, max_seqs=64,
                   use_async=True, native_policy="fcfs", native=False, **policy):
    """A real vLLM scheduler with the plugin (``mode``: vtc or passthrough), or without it (``native``)."""
    weights = weights or {"A": 1.0, "B": 1.0}
    out = Path(tempfile.mkdtemp(prefix="fairsched-test-"))
    (out / "policy.json").write_text(json.dumps(
        {"tenants": {t: {"weight": w} for t, w in weights.items()}, **policy}))
    os.environ.update(FAIRSCHED_MODE=mode, FAIRSCHED_POLICY=str(out / "policy.json"), FAIRSCHED_OUTPUT=str(out))
    model = ModelConfig(model=MODEL, dtype="float16", seed=0, skip_tokenizer_init=True, max_model_len=16384)
    sched = SchedulerConfig(max_num_seqs=max_seqs, max_num_batched_tokens=budget, max_model_len=16384,
                            is_encoder_decoder=False, enable_chunked_prefill=True, policy=native_policy,
                            async_scheduling=use_async)
    cache = CacheConfig(block_size=BLOCK, gpu_memory_utilization=0.9, cache_dtype="auto",
                        enable_prefix_caching=True)
    # No GPU in the test container, so the platform cannot be inferred.
    config = VllmConfig(scheduler_config=sched, model_config=model, cache_config=cache,
                        device_config=DeviceConfig(device="cpu"))
    kv = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32))])
    cache.num_gpu_blocks = num_blocks
    if native:
        cls = AsyncScheduler if use_async else Scheduler
    else:
        cls = TenantAsyncScheduler if use_async else TenantSyncScheduler
    return cls(vllm_config=config, kv_cache_config=kv, structured_output_manager=StructuredOutputManager(config),
               block_size=BLOCK, log_stats=False)


def make_request(tenant, prompt, max_tokens=8, *, tokens=None, priority=0):
    n = next(_ids)
    rng = random.Random(n)
    ids = tokens if tokens is not None else [rng.randrange(1000, 30000) for _ in range(prompt)]
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True,
                            extra_args={"fairsched_tenant": tenant, "fairsched_rid": f"{tenant}-{n}"})
    return Request(request_id=f"{tenant}-{n}", prompt_token_ids=ids, sampling_params=params,
                   pooling_params=None, arrival_time=float(n), priority=priority, block_hasher=_HASHER)


class Engine:
    """Minimal EngineCore loop: async keeps one step in flight, like vLLM's batch queue."""

    def __init__(self, scheduler):
        self.s = scheduler
        self.inflight: deque = deque()
        self.step = 0
        self.first_scheduled: dict[str, int] = {}
        self.first_token: dict[str, int] = {}
        self.preempted: list[str] = []
        self.finished: dict[str, int] = {}

    def add(self, request):
        self.s.add_request(request)

    def run_step(self):
        self.step += 1
        out = self.s.schedule()
        self.preempted += [t for t, n in getattr(self.s, "fairsched_preempted", {}).items() for _ in range(n)]
        samples = {}
        for rid in out.num_scheduled_tokens:
            self.first_scheduled.setdefault(rid, self.step)
            samples[rid] = not self.s.requests[rid].is_prefill_chunk
        self.inflight.append((out, samples))
        depth = 1 if isinstance(self.s, AsyncScheduler) else 0
        while len(self.inflight) > depth:
            self._complete(*self.inflight.popleft())
        return out

    def drain(self):
        while self.inflight:
            self._complete(*self.inflight.popleft())

    def _complete(self, out, samples):
        rids = list(out.num_scheduled_tokens)
        for rid in rids:
            if samples[rid]:
                self.first_token.setdefault(rid, self.step)
        runner = ModelRunnerOutput(req_ids=rids, req_id_to_index={r: i for i, r in enumerate(rids)},
                                   sampled_token_ids=[[7] if samples[r] else [] for r in rids])
        before = set(self.s.requests)
        self.s.update_from_output(out, runner)
        for rid in before - set(self.s.requests):
            self.finished.setdefault(rid, self.step)

    def run_until_idle(self, limit=10_000):
        while (self.s.requests or self.inflight) and self.step < limit:
            if self.s.requests:
                self.run_step()
            else:
                self.drain()
        return self.step

    def abort(self, request_id):
        self.s.finish_requests(request_id, RequestStatus.FINISHED_ABORTED)
