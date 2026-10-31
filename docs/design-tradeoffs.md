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
other tenant has nothing to send. A per-tenant concurrency cap at the gateway
(16 in flight, one value tried, on the noisy-neighbour workload) kept the
interactive tenant's median TTFT at 0.09–0.10 s with no preemptions, better
than VTC, but its p99 was 2.05–2.94 s, and capping the heavy tenant's
concurrency delayed the end of its burst from 114–118 s to 142–147 s. VTC split the aggressive workload 0.50 :
0.50, gave 2 : 1 weights 0.67 : 0.33, kept the interactive tenant's median
TTFT at 0.39–0.50 s through the burst, and gave the remaining tenant
everything once the other ran out of work.

**Cost.** When the interactive tenant is known, priority does better
(0.23–0.30 s median, 0.70–0.78 s p99 against VTC's 0.86–1.43 s). Alternating
between backlogged tenants churns the KV cache, so throughput is 0.5–1.1%
lower in the aggressive run and 2.0% lower with weights, matching the extra
recompute.

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
rerun. The hooks add 132, 244 and 403 us of CPU per step at 64, 128 and 256
running requests, about 0.5% of a 50 ms decode step at 128, and with async
scheduling that time overlaps the previous step's GPU work. End to end, on a
saturating trace, the plugin in passthrough mode delivers 0.1–0.5% less
throughput than vLLM without it (per seed).

## Never preempt for fairness

**Decision.** Fairness acts only through admission order and the order in
which a step's token budget is spent. Requests are preempted only when vLLM
itself runs out of KV blocks.

**Alternatives.** Preempt a running request whose tenant gets too far ahead,
as an earlier version did.

**Why.** That version livelocked on a deliberately overloaded run with
6,144-token prompts: no request finished, and in the part of the run its
step log covers it preempted 678 times, 99.5% of the prefill tokens it
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
requests from preemption.

**Why.** It adds no decision point: vLLM's preemption path, and its
correctness, stay as they are, and an over-served tenant whose request misses
the budget waits a step without losing its KV.

**Cost.** The most-served tenant at a given moment is not always the heavy
one. Right after its counter is lifted (next section), the light tenant can
lead, and its newest request is then the victim: the interactive tenant lost
8–20 requests this way in the noisy-neighbour runs (0–8 under FCFS, 0 under
priority). That may explain its p99 above priority's; the summaries count
preemptions but do not tie them to particular slow requests.

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
model attention growing with context length.

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

## Counters in memory, for one engine

**Decision.** Counters and accounts live in the engine process.

**Alternatives.** Persist them, or share them across replicas.

**Why.** Fairness is defined over one engine's lifetime, and persistence
would put I/O into the scheduling step.

**Cost.** Counters reset when the engine restarts, and there is no fairness
across replicas or GPUs; multi-GPU, speculative decoding, LoRA and KV
connectors are refused at startup.

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
saturating `aggressive` trace (`results/overhead/`): 0.1–0.5% lower
throughput per seed. Open-loop traces cannot show it, because their arrival
schedule sets the run length (every arm's seed-1 noisy-neighbour run gives
730.6–731.2 output tokens/s).

**Cost.** The full set of runs takes about 3 hours.
