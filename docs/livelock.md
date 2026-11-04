# Case study: a fairness policy that livelocked long prompts

An earlier design of this scheduler did more than reorder requests. When a
running request's tenant got too far ahead on the virtual counter, the
scheduler **preempted the request itself** to hand its batch slot to someone
else. On a deliberately overloaded workload of long prompts this produced a
livelock: the GPU stayed busy for the whole run, and no request produced a
single output token.

This page covers what happened, why it happened, and what the current design
does instead. The earlier design was replaced by the current one and is not
part of this repository; the outline below shows the part that mattered.

**What this page rests on.** Every count here comes from one committed file,
[results/livelock/summary.json](../results/livelock/summary.json), which
`scripts/livelock_summary.py` computed from the run's step log. The step log,
the client's output and the earlier design's code are not in this repository;
the file records the SHA-256 of the log and of that code. So the counts
cannot be recomputed and the run cannot be repeated from here. Besides the
counts, the file records the model, the step budget, the context length,
prefix caching, the client timeout, the KV blocks, the policy settings and
each request's final status. The rest of the setup below (GPU, vLLM version,
eager and synchronous mode, the four-sequence limit, prompt and output
lengths), the plain-FCFS failure and the code outline under [Why](#why) are
not backed by any file in this repository.

## What we saw

The run used Qwen2.5-1.5B on an RTX 4090 with vLLM 0.31.0: synchronous
scheduling in eager mode, no prefix caching, a 512-token per-step budget
(`--max-num-batched-tokens 512`), at most 4 running sequences, and a
deliberately small KV cache of 512 usable blocks (8,192 tokens). There were
three tenants with equal weights. Eight requests had 6,144-token prompts and
up to 2,048 output tokens. One request had a 256-token prompt.

On the client side, **0 of 9** requests completed: all hit the 90 s client
timeout, and no output token arrived.

The scheduler's step log has a record for each of the run's 2,036 steps,
from the first to the abort of all nine requests at 90.0 s. Its writer
flagged an error in its footer, which is why the log was earlier described
as partial; the step records themselves are contiguous. Over those steps:

| | Result |
|---|---|
| Policy-initiated preemptions | 678 |
| Prefill tokens dispatched for the 3 requests that were scheduled | 347,136 / 347,648 / 347,136 |
| Unique prompt positions those 3 ever reached | 1,024 / 2,048 / 2,048 |
| Dispatched prefill tokens that re-executed a position already computed | 99.5% (1,036,800 of 1,041,920) |
| Decode tokens | 0 |

These are token counts, not GPU-time measurements. vLLM itself preempted
nothing; all 678 preemptions were the policy's.
[results/livelock/summary.json](../results/livelock/summary.json) holds the
counts per request, the policy settings the log records (`w_p = w_d = 1`,
`charge_recompute = "tenant"`, two guaranteed dispatches, 513 KV blocks
including vLLM's null block), the SHA-256 of the log's files and index, and
the SHA-256 of the design's `scheduler.py` and of its whole package
(policy version `p1-original9-vtc-minimum2-v1`).
`scripts/livelock_summary.py` computed it from the run directory. The other
six requests were never scheduled; the three that were held the KV cache.

The workload was overloaded by construction: one request's full 8,192-token
context needed the entire KV cache. Plain FCFS also failed it with these
limits, with requests hitting the same timeout, and there are no step-level
numbers for that FCFS run to compare with. So the run does not show that the
policy made a feasible workload infeasible. What it shows is where the quantum
policy's work went: almost all of the logged prefill work recomputed
positions that had already been computed once, and nothing reached decode.

## Why

The earlier design's `_apply_service_quantum` ran on every step. In outline:

```python
quantum = max_num_scheduled_tokens * max(w_p, w_d) / min_weight     # = 512 tokens here
ceiling = floor + quantum + delta                                     # floor = least-served tenant
victims = [r for r in running if counter[tenant(r)] > ceiling]
for r in victims:
    Scheduler._preempt_request(r)        # frees the request's KV blocks
```

Three properties combined into a livelock:

1. **The fairness window was smaller than one request.** A 6,144-token prompt
   needs 12 prefill chunks at 512 tokens per step. After one or two chunks
   (two steps were guaranteed by a minimum-service setting), its tenant's
   counter was more than 512 above the floor, so the request was preempted.
2. **Preemption threw the work away.** vLLM's `_preempt_request` frees the
   request's KV blocks and resets `num_computed_tokens` to 0. The run had no
   prefix caching, so none of the freed work could be reused when the request
   came back. Every resumption started from scratch.
3. **The victim also paid for the recompute.** With `charge_recompute="tenant"`,
   redoing those 1,024–2,048 tokens raised the victim tenant's counter again.
   That pushed it straight back over the ceiling the next time it got a chunk.

Each tenant took turns running for one or two chunks and then being thrown
out. The counters stayed balanced, which is exactly what the policy was built
to guarantee, while useful throughput was zero. Strict fairness had become
more important than work conservation.

The general lesson: **a fairness mechanism that destroys work must not
enforce a bound tighter than the largest unit of work it can interrupt.** VTC's
own fairness bound already contains a term for the maximum request size. That
term is the price of not interrupting requests, and you cannot remove it by
preempting more often.

## What the scheduler does now

The current plugin never preempts anything itself. Fairness comes from two
levers that never destroy work:

* **Admission order.** The waiting queue releases the oldest request of the
  least-served tenant first (`TenantRequestQueue`).
* **Token-budget order.** Before vLLM's native `schedule()` runs, the running
  list is sorted least-served first. vLLM spends the step's token budget in
  that order, so an over-served tenant's running requests wait a step instead
  of losing their KV.

Preemption still happens, but only when vLLM itself runs out of KV blocks.
vLLM then evicts `running[-1]`, which after the sort is the newest request of
the most over-served tenant. Recompute caused by that eviction is billed to a
`system` account by default, so a tenant is not charged for a preemption it
did not cause.

## Regression coverage

* `tests/test_scheduler_vllm.py::test_long_prompts_finish_under_kv_pressure`
  runs the real vLLM 0.31.0 scheduler on CPU with three 6,144-token prompts
  next to ten short requests, a 512-token budget and a 450-block KV cache. It
  asserts that vLLM's own preemption does fire and takes only the over-served
  tenant, that every request finishes, that no request is preempted more than
  twice, and that the run takes fewer steps than three times the useful work
  divided by the budget (plus 96).
* `tests/test_scheduler_vllm.py::test_kv_pressure_preempts_the_most_overserved_tenant`
  checks that when KV does run out, only the over-served tenant is evicted,
  whereas native FCFS evicts the newest request regardless of tenant.
* The `long_prompt` GPU workload (`results/long_prompt/`) uses the same
  6,144-token prompt length on Qwen2.5-7B with a 520-block (8,320-token) KV
  cache. The earlier design was not run on that configuration. See the README
  for the measured result.
