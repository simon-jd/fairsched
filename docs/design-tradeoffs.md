# Design trade-offs

fairsched comes down to a few choices about where fairness lives: inside the
scheduler or in front of it, by reordering or by preempting, billed when work
is scheduled or when it completes. Each section records one of them with the
options it was weighed against and the price it pays, measured where a
measurement exists. Numbers are from the README's repeated-seed runs (Qwen2.5-7B
on one RTX 4090, vLLM 0.31.0); ranges are over seeds.

## Share the GPU by service received

**Decision.** Each tenant has a Virtual Token Counter of weighted service
received, and admission serves the backlogged tenant with the smallest one
(VTC, Sheng et al., OSDI 2024).

**Alternatives.** vLLM's FCFS; vLLM's static priorities; rate limits,
quotas or concurrency caps at the gateway.

**Why.** FCFS serves a tenant in proportion to what it sends: with three
times the demand, the aggressive tenant took 75% of the GPU, and a 300-request
burst pushed the interactive tenant's median time to first token to
28.8–40.5 s. Static priority needs to know who is interactive and is all or
nothing between peers (1.00 : 0.00). Rate limits leave the GPU idle when the
other tenant has nothing to send. A per-tenant concurrency cap at the gateway,
swept at 4, 8, 16 and 32 in flight on the noisy-neighbour workload, preempted
nothing at any value. At 32 it beat VTC on both tenants' median TTFT (the
interactive tenant's 13–63% lower) with the burst finishing within 2.1 s of
VTC's, but the interactive tenant's p99 was 4.17–6.06 s against VTC's
0.86–1.43 s. At 16 that tenant's median was 0.09–0.15 s and its p99
1.67–2.98 s, and capping the heavy tenant delayed the end of its burst from
113–118 s to 142–147 s; at 8 and 4 the burst ended after 194 s and 313 s.
VTC split the aggressive workload 0.50 : 0.50, gave 2 : 1 weights
0.67 : 0.33, kept the interactive tenant's median TTFT at 0.39–0.50 s
through the burst, and gave the remaining tenant everything once the other
ran out of work.

**Cost.** When the interactive tenant is known, priority does better
(0.23–0.30 s median, 0.70–0.78 s p99 against VTC's 0.86–1.43 s). Alternating
between backlogged tenants churns the KV cache, so throughput is 0.53% lower
in the aggressive workload (95% interval 0.23–0.83%, six seeds with the run
order rotated) and 2.0% lower with weights, matching the extra recompute.

## A scheduler subclass, pinned to one vLLM version

**Decision.** The plugin subclasses vLLM's `AsyncScheduler` and loads with
`--scheduler-cls`. It refuses to start on any vLLM but 0.31.0, and the CPU
tests run the real 0.31.0 scheduler, KV cache manager and prefix cache with
only the model forward replaced.

**Alternatives.** Reorder requests in a proxy in front of vLLM; fork vLLM.

**Why.** Once a request is in the engine's queue, a proxy can no longer
reorder it, and it never sees the step loop, the running set or KV pressure,
where the decisions are made. A subclass changes the decisions without
changing vLLM, which runs from its official container.

**Cost.** The hooks depend on internals (`running`, `waiting`, hook names,
`num_computed_tokens` semantics), so each vLLM upgrade needs the CPU tests
rerun. With VTC the plugin adds 172, 311 and 674 us of CPU per step at 64,
128 and 256 running requests (medians of 5 runs, `results/hook_overhead.jsonl`),
about 0.6% of a 50 ms decode step at 128, and with async
scheduling that time overlaps the previous step's GPU work. End to end, on a
saturating trace with the run order rotated over six seeds, passthrough
mode's throughput is within 0.11% of vLLM without the plugin, seed for seed,
and the mean gap (−0.035%, 95% interval −0.10% to +0.03%) is not resolved
from zero.

## Never preempt for fairness

**Decision.** Fairness acts only through admission order and the order in
which a step's token budget is spent. Requests are preempted only when vLLM
itself runs out of KV blocks.

**Alternatives.** Preempt a running request whose tenant gets too far ahead,
as an earlier version did.

**Why.** That version livelocked on a deliberately overloaded run with
6,144-token prompts: no request finished, and over the run's 2,036 logged
steps it preempted 678 times, 99.5% of the prefill tokens it
dispatched were recompute, and no decode token ran
([livelock.md](livelock.md)). FCFS also timed out on that workload, so the
run shows a mechanism rather than a measured regression: the fairness window
was smaller than one request, and preemption throws the request's KV away.
On 6,144-token prompts in the `long_prompt` workload (a different model and
KV size, on which the earlier version was not run) the current design
finishes the long requests as fast as FCFS (16.1–16.3 s) with no
preemptions, and the short tenant's p99 TTFT is 1.9–3.0 s instead of
10.3–12.8 s.

**Cost.** A request that is already running a long prefill keeps its chunks
going; the fairness bound includes the largest request, as VTC's own bound
does.

## Steer vLLM's own preemption by the order of `running`

**Decision.** Before the native `schedule()`, `running` is sorted
least-served first. The step's budget goes to under-served tenants first,
and `running[-1]`, vLLM's victim under KV pressure, is the newest request of
the most-served tenant.

**Alternatives.** Choose victims in the plugin; exempt short or interactive
requests from preemption; order the tail by KV held instead of service (now
the `kv_share` option).

**Why.** It adds no decision point: vLLM's preemption path, and its
correctness, stay as they are, and an over-served tenant whose request misses
the budget waits a step without losing its KV.

**Cost.** The most-served tenant at a given moment is not always the heavy
one. Right after its counter is lifted (next section), the light tenant can
lead, and its newest request is then the victim: the interactive tenant lost
8–20 requests this way in the noisy-neighbour runs (0–8 under FCFS, 0 under
priority). The `kv_share` order, which sends tenants over their weighted
share of the running KV to the back, brought that to 0 in the same session
(against 7–15), while the heavy tenant lost 4–8 more requests per seed. The
interactive tenant's TTFT did not improve, so these preemptions are not what
keeps its p99 above priority's. In the aggressive workload `kv_share` was
faster in each of six seeds with the order alternated, by 0.68% on average
(95% interval 0.13–1.23%); its two largest gaps came with a higher GPU clock
in the `kv_share` run. It stays an option rather than the default because
it moves preemptions to the heavy tenant without a measured latency gain.

## Lift a returning tenant's counter

**Decision.** A tenant that becomes backlogged again starts at the smallest
counter among backlogged tenants, or at the last departing tenant's counter
when none is backlogged.

**Alternatives.** Keep its old counter; reset it to zero.

**Why.** Without the lift a tenant could bank idle time: after an hour away it
would monopolise the GPU until its counter caught up. This is the paper's
rule, and its fairness bound depends on it.

**Cost.** The returning tenant restarts level with the busiest one, which,
with the victim order above, is what exposes it to preemption.

## Bill the work that runs, when it is scheduled

**Decision.** `_update_after_schedule` charges each tenant for the positions
vLLM scheduled in the step: prefill at weight 1, decode at weight 2. Cached
prefix positions are never scheduled and so never billed. Re-executed
positions after a preemption are detected from each request's set of executed
positions and billed to a `system` account. Work for requests that ended
while their step was on the GPU is marked wasted. Each admission also
reserves its request's remaining prefill cost on the tenant's counter for the
rest of the step.

**Alternatives.** Bill a request's prompt and output tokens on admission or on
completion.

**Why.** With chunked prefill, prefix caching, preemption and cancellation,
the work a request causes differs from its token counts, and billing at
completion lets a long request run unbilled for its whole life. Without the
reservation, one tenant's queued requests could all be admitted in one step
before any was billed. Counters (which drive decisions) and accounts (who
paid) are separate, so reclassifying waste never rewrites decisions already
made.

**Cost.** Per-request interval sets and a ledger per in-flight step, released
only when no step still refers to them. The cost model is linear and does not
model attention growing with context length, and its weights are not this
GPU's. A step-time fit on the logged work (R² 0.977–0.998) puts a decode
token at 0.97–1.71× a prefill token, depending on the workload, against
VTC's 2×. With a prefill-heavy and a decode-heavy tenant, VTC's even billed
split gave the prefill-heavy one 53–54% of the fitted GPU time.

Billing decode at the fitted ratio did not fix that. With `w_d = 0.97` the
GPU-time split on the mixed workload swung past even to 0.40 : 0.60,
because a marginal weight leaves out the fixed 12.7–14.6 ms each of its steps
costs, and the interactive
tenant in the noisy-neighbour workload got slower (median TTFT 0.51–0.56 s
against 0.39–0.53 s, 21–23 preemptions against 8–25). The default stays 2.

The plugin does not bill the fitted attention term (context length). Its
coefficient moved tenfold between workloads (4.8–48 ns per query-key pair),
so a billed value would be wrong for the next workload. It would also add a
third term to every charge path: the admission reservation would have to
project attention for prompt chunks not yet scheduled, and the recompute and
waste reclassification would have to carry it too.

## Tenant identity from the gateway, never from the client

**Decision.** A Go gateway maps API keys to tenants. It strips every
client-supplied `fairsched*`, `vllm_xargs` and `priority` field from the top
level of the body (and a top-level `extra_body`), where vLLM reads them,
injects a trusted `vllm_xargs`, always streams, and cancels the
upstream request when the client disconnects.

**Alternatives.** Trust a tenant field in the request; map keys inside vLLM's
API server.

**Why.** A client must not be able to claim another tenant's identity or a
priority. Cancellation has to reach vLLM's abort path, so the engine stops
work nobody will read and the plugin can account it as wasted.

**Cost.** One more streaming hop, and the key-to-tenant map is static
configuration.

## Counters in memory, shared across replicas only when asked

**Decision.** Counters and accounts live in the engine process. With
`FAIRSCHED_COUNTER_URL` set, a thread in each engine posts the engine's
cumulative billed service per tenant to a small counter service every
`FAIRSCHED_SYNC_MS` (100 ms by default) and reads back the other replicas'
totals. Ordering, lifts and victim choice then use the engine's own counter
plus the others' totals. Without the URL nothing changes: one engine, local
counters.

**Alternatives.** Persist the counters; a central scheduler that admits for
every replica; a synchronous exchange inside `schedule()`; publishing
increments instead of totals.

**Why.** Fairness is defined over one engine's lifetime, and persistence
would put I/O into the scheduling step. Per-replica counters break down when
routing has affinity. With A's sessions all on one of two engines and B
spread over both, each engine splits its own service evenly, so A got 0.27
of the global service instead of 0.50
([Two replicas](../README.md#two-replicas-per-replica-and-shared-counters)).
The exchange is off the scheduling thread: each step only swaps in the
latest totals the sync thread fetched. Cumulative totals make a lost or
repeated post harmless, and a restarted service is whole again after one
round. Lifts stay local, because only billed service is published.

**Cost.** Measured on two A10-24Q vGPUs in one Snowflake node. Counters
still reset when an engine restarts, and the others' totals are up to one
period (plus a round trip) old. The step pays 1.2–4.1 µs at the median to
swap totals, against a `schedule()` median of 0.54–0.83 ms; the HTTP round
trip runs on the sync thread. If the service is unreachable, the engine logs
it once and keeps scheduling on its own counters plus the others' last known
totals, so fairness falls back to per-replica: with the service killed
mid-run, A's share went from 0.50 to 0.25. Shared counters move service; they
do not create it. The engine that holds a tenant's affinity traffic serves
that tenant almost alone while it is behind, so the other tenant's requests
on that engine wait longer: B's median TTFT there was 69.9–70.2 s, against
62.3–63.5 s with per-replica counters. The service is one process with no
replication, tested with two engines in one node. Multi-GPU engines,
speculative decoding, LoRA and KV connectors are still refused at startup.

## Log every step, off the scheduling thread

**Decision.** Each step appends one JSON line (per-tenant tokens, counters,
accounts, preemptions, free KV blocks, `schedule()` time) through a bounded
queue and a writer thread.

**Alternatives.** Log synchronously; log only aggregates.

**Why.** The service timelines and every number in the README come from these
lines, and disk I/O must never stall `schedule()`.

**Cost.** If the writer falls behind, records are dropped and counted rather
than delaying the engine.

## Measure with open arrivals and a passthrough control

**Decision.** The load generator sends each request at its scheduled time
whether or not earlier ones have finished, and measures TTFT from that time.
The FCFS arm runs the plugin in passthrough mode, and every arm runs three
seeds (five for the two headline workloads).

**Alternatives.** Closed-loop clients; vLLM without the plugin as the
control; one seed.

**Why.** A closed loop slows its own arrivals when the server queues, which
hides the queueing delay this project is about. With passthrough, logging and
accounting cost the same in both arms, so a difference is the policy's. What
passthrough itself costs against vLLM without the plugin is measured
separately, on the CPU (`bench/hook_overhead.py`) and end to end on the
saturating `aggressive` trace (`results/overhead_rotated/`): within 0.11%
per seed, not resolved from zero. Open-loop traces cannot show it, because their arrival
schedule sets the run length (every arm's seed-1 noisy-neighbour run gives
730.6–731.2 output tokens/s).

**Cost.** The full set of runs takes about 5 hours.
