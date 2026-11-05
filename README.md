# fairsched: fair multi-tenant scheduling for vLLM

**Problem.** Tenants that share one vLLM engine are served in proportion to
what they send. On one GPU serving one model (Qwen2.5-7B), with synthetic
seeded traces, a 300-request burst from one tenant under vLLM's FCFS pushed
another tenant's median time to first token (TTFT) to 29–41 s, and a tenant
sending three times the other's demand took 75% of the service. Static priority
helps only if you know in advance who is interactive; between peers it gives
one tenant everything.

**Approach.** fairsched is an out-of-tree scheduler plugin for vLLM 0.31.0
that implements the Virtual Token Counter (VTC; Sheng et al., *Fairness in
Serving Large Language Models*, OSDI 2024). It orders admission and each
step's token budget by the weighted service each tenant has received, bills
only the tokens that actually ran, and never preempts a request on its own.
A small Go gateway maps API keys to tenants.

**Scope:** one RTX 4090; Qwen2.5-7B-Instruct BF16 on vLLM 0.31.0 (vLLM is
not modified); synthetic, seeded two-tenant traces.

Ranges are over seeds; bold marks the better side:

| Workload and measure | VTC | Baseline | Seeds |
|---|---|---|---|
| Two backlogged peers, B sends 3× A's demand: A's share of billed service | **0.50** | 0.25, FCFS | 5 |
| 300-request burst from B: A's median TTFT | **0.39 – 0.50 s** | 28.8 – 40.5 s, FCFS | 5 |
| Same burst: A's median TTFT | 0.39 – 0.50 s | **0.23 – 0.30 s**, static priority, A marked high | 5 |
| Same burst: A's median TTFT | 0.39 – 0.50 s | **0.15 – 0.36 s**, gateway cap of 32 requests per tenant | 5 |
| Same burst: A's p99 TTFT | **0.86 – 1.43 s** | 4.17 – 6.06 s, gateway cap of 32 | 5 |
| Saturating trace: output tokens/s, median over seeds | 1,311 | **1,317**, FCFS | 6, arm order rotated |

Rows 1–3 come from `results/aggressive/summary.json` and
`results/noisy_neighbor/summary.json`, rows 4–5 from
`results/cap_sweep/summary.json` (against the same seeds' VTC runs), row 6
from `results/overhead_rotated/paired.json`. FCFS is vLLM's scheduler with
the plugin in passthrough mode.

**Limitations**
- Not the fastest option for the interactive tenant (rows 3–5). Static
  priority has to be told who is interactive. A cap of 32 also gives B a
  1.7–3.0 s lower median TTFT than VTC; VTC wins only on A's tail. Smaller
  caps delay B's burst by 28 s or more ([cap sweep](#noisy-neighbour)).
- Throughput is 0.53% below FCFS on a saturating trace (paired over 6 seeds,
  95% interval 0.23–0.83%), with more preemption recompute.
- The light tenant is preempted under KV pressure: 8–20 of A's requests per
  burst run, against 0–8 under FCFS. The optional `kv_share` victim order
  brings that to 0 but does not lower A's latency.
- VTC's billing weights are not this GPU's costs. With a prefill-heavy and a
  decode-heavy tenant, an even billed split gave the prefill-heavy one 53–54%
  of the fitted GPU time ([request shapes](#different-request-shapes-and-what-a-token-costs)).
- One GPU, one model, synthetic traces, pinned to vLLM 0.31.0
  ([all limitations](#limitations)).

[How it works](#how-it-works) · [Design trade-offs](docs/design-tradeoffs.md) ·
[Full results](#results) · [Livelock post-mortem](docs/livelock.md) ·
[Reproduce](#reproduce)

## What is in this repository

The work here is the integration and the evaluation:
- where to hook a continuous-batching scheduler that has chunked prefill,
  async scheduling, prefix caching and KV-pressure preemption, without
  modifying vLLM;
- how to bill only the work that actually ran;
- a post-mortem of an earlier design that livelocked;
- repeated-seed experiments on an RTX 4090 that report VTC's costs as well as
  its wins, against vLLM's FCFS and priority policies and a gateway
  concurrency cap.

## How it works

The plugin subclasses vLLM's `AsyncScheduler` and loads with
`--scheduler-cls fairsched.TenantAsyncScheduler`. It hooks the scheduler in
three places ([docs/design.md](docs/design.md); the decisions and their costs
are in [docs/design-tradeoffs.md](docs/design-tradeoffs.md)):

1. **Admission.** `self.waiting` becomes a per-tenant queue. Each pop returns
   the oldest request of the least-served tenant, the one with the lowest
   virtual counter. A tenant that returns after its queue emptied is lifted
   to the lowest counter among backlogged tenants, as in the paper, so idle
   time cannot be banked.
2. **Token budget and preemption victim.** Before vLLM's native `schedule()`
   runs, the running list is sorted least-served first.
   - vLLM spends each step's token budget in that order.
   - Under KV pressure vLLM preempts `running[-1]`. That is now the newest
     request of whichever tenant is most over-served at that moment, which
     can be the light tenant right after a lift (see the costs below). The
     `kv_share` option takes the victim from a tenant over its share of the
     KV instead ([below](#choosing-the-preemption-victim-by-kv-share)).
3. **Billing.** `_update_after_schedule` charges each tenant, before positions
   advance, for the tokens vLLM actually scheduled that step:
   - prefill tokens at weight 1, decode tokens at weight 2;
   - prefix-cache hits are never billed;
   - recompute after a preemption is detected from executed positions and
     billed to a separate `system` account;
   - steps of requests aborted while their step was on the GPU are
     reclassified as wasted.

```mermaid
flowchart LR
  C[clients] -->|API key| G[Go gateway<br/>key → tenant]
  G -->|vllm_xargs: tenant| V[vLLM API server]
  V --> S[TenantAsyncScheduler]
  S -->|least-served head| W[(waiting: per-tenant queues)]
  S -->|sorted least-served first| R[(running)]
  S -->|bill scheduled tokens| L[(counters + accounts)]
```

The plugin never preempts a request on its own. An earlier design did, and
on a deliberately overloaded run with 6,144-token prompts (Qwen2.5-1.5B,
512-token step budget) it **livelocked**:
- no request produced a token before the 90 s client timeout;
- over the run's 2,036 logged steps the policy preempted 678 times, and
  99.5% of the prefill tokens it dispatched recomputed positions already
  computed once.

**This claim rests on one committed file,
[results/livelock/summary.json](results/livelock/summary.json).** It holds
counts computed from that run's step log, plus the SHA-256 of the log and of
the earlier design's source. The step log and that design's code are not in
this repository, so the counts cannot be recomputed or rerun from here. The
prompt length is not recorded in that file.

Plain FCFS also failed that workload: with the same limits its requests hit
the same timeout. No data from that FCFS run is in this repository either.
So the run shows where the policy's work went, not that FCFS would have
finished. [docs/livelock.md](docs/livelock.md) explains why enforcing
fairness below the size of one request destroys work.

## Results

**Setup**
- One RTX 4090, Qwen2.5-7B-Instruct BF16, vLLM 0.31.0 in its official
  container, with async scheduling and prefix caching.
- Engine limits: `--max-num-batched-tokens 2048`, `--max-num-seqs 128`, and
  84k tokens of KV.
- Load: the Go load generator replays seeded open-arrival traces through the
  gateway. Prompts are exact-length token IDs, and `ignore_eos` fixes output
  lengths.
- `noisy_neighbor` and `aggressive` ran with 5 seeds per arm, the
  gateway-cap sweep with 5, the rotated-order overhead and `kv_share` reruns
  with 6, the other workloads and arms with 3
  ([run history](results/README.md#run-history)). In
  `noisy_neighbor` and `long_prompt` the
  seed changes tenant A's Poisson arrival times (and, in `noisy_neighbor`,
  which of B's requests are cancelled). In `aggressive` and `weighted` the
  arrival schedule is fixed and the seed changes only the prompt token IDs.
  Tables give the range over seeds.
- Percentiles are nearest-rank over each run's completed requests. Where a
  tenant has fewer than 100 requests (A in `long_prompt` has 63–69), p99 is
  the maximum.
- Arms:
  - **FCFS:** vLLM's scheduler with the plugin in passthrough mode, so
    accounting and logging match the VTC arm;
  - **FCFS + gateway cap** (noisy neighbour and mixed shapes): FCFS as
    above, with the gateway admitting at most 16 requests per tenant to the
    engine and queueing the rest (its `max_inflight`). 16 of B's requests
    hold about 37k tokens, under half of the KV cache. On the noisy-neighbour
    trace the cap was also run at 4, 8 and 32;
  - **priority:** vLLM's native priority policy, with the gateway marking one
    tenant high;
  - **VTC;**
  - **no plugin** (overhead runs only): vLLM's FCFS without the plugin.
- "Share" is each tenant's fraction of billed service over the steps in
  which both tenants had work waiting.
- Per-run data is in `results/*/summary.json`
  ([results/README.md](results/README.md)). To rerun, see
  [Reproduce](#reproduce).

### Noisy neighbour

**Workload.**
- Tenant A: Poisson 2 req/s, 256-token prompts, 128 output tokens, for 120 s.
- Tenant B: at t = 10 s, 300 requests with 2,048-token prompts and 256 output
  tokens. The client cancels a random 20% of them after their second output
  chunk.

| | FCFS | FCFS + gateway cap | Priority (A high) | VTC |
|---|---|---|---|---|
| A TTFT median | 28.8 – 40.5 s | 0.09 – 0.15 s | 0.23 – 0.30 s | 0.39 – 0.50 s |
| A TTFT p99 | 89.5 – 92.0 s | 1.67 – 2.98 s | 0.70 – 0.78 s | 0.86 – 1.43 s |
| B TTFT median | 41.0 – 46.3 s | 64.2 – 65.1 s | 48.0 – 50.2 s | 48.4 – 50.2 s |
| B's burst finishes | 112 – 115 s | 142 – 147 s | 113 – 116 s | 113 – 118 s |
| A requests preempted under KV pressure | 0 – 8 | 0 | 0 | 8 – 20 |

The gateway-cap column is the cap-16 arm of the sweep below
(`results/cap_sweep/`), on the same traces.

![Tenant A TTFT CDF, noisy neighbour, seed 1](results/noisy_neighbor/ttft_A.png)

Under FCFS, A gets almost no service from t = 10 s until B's backlog drains
near t = 100 s. Under VTC, A is served at its demand throughout. B pays with
2.4–8.0 s more median TTFT than under FCFS, and its burst finishes 0.6–2.7 s
later (same seed):

![Billed service per tenant over time](results/noisy_neighbor/service.png)

**Static priority** with A marked high is faster for A than VTC in every
seed. A's median TTFT is 1.5–1.8× lower (0.23–0.30 s), and its p99 is lower
too (0.70–0.78 s). It has to be told which tenant is interactive, and between
backlogged peers it gives one tenant everything
([aggressive tenant](#aggressive-tenant)).

**A concurrency cap at the gateway** is the obvious alternative to a
scheduler. With each tenant limited to 16 requests in the engine, it beats
VTC on two counts: A's median TTFT is 0.09–0.15 s, 2.6–5.4× lower than VTC's
with the same seed, and nothing is preempted. It loses on the others. A's
p99 is 1.67–2.98 s, 1.8–2.6× VTC's. With at most 16 of B's requests running,
B's median TTFT rises to 64.2–65.1 s and its burst finishes at 142–147 s,
28–33 s later than under VTC with the same seed, so the same work takes
longer (611–660 output tokens/s over the run, against 711–802 for the other
arms).

**Other cap values.** The cap was swept at 4, 8, 16 and 32 requests per
tenant on seeds 1–5, the seeds VTC ran, in one session with the arm order
rotated per seed (`results/cap_sweep/`). The FCFS and VTC columns are the
`noisy_neighbor/` runs, made separately
([run history](results/README.md#run-history)). The sweep's cap-16 runs
reproduce the first cap-16 runs of seeds 1–3 (`results/noisy_neighbor/`) to
within 0.01 s of A's median TTFT, 0.17 s of its p99 and 0.24 s of the
burst's finish.

| | Cap 4 | Cap 8 | Cap 16 | Cap 32 | No cap (FCFS) | VTC |
|---|---|---|---|---|---|---|
| A TTFT median | 7.2 – 27.5 s | 0.08 – 0.37 s | 0.09 – 0.15 s | 0.15 – 0.36 s | 28.8 – 40.5 s | 0.39 – 0.50 s |
| A TTFT p99 | 21.9 – 53.0 s | 2.23 – 5.04 s | 1.67 – 2.98 s | 4.17 – 6.06 s | 89.5 – 92.0 s | 0.86 – 1.43 s |
| B TTFT median | 149.6 – 161.4 s | 91.6 – 96.4 s | 64.2 – 65.1 s | 46.5 – 47.2 s | 41.0 – 46.3 s | 48.4 – 50.2 s |
| B's burst finishes | 313 – 334 s | 194 – 206 s | 142 – 147 s | 113 – 115 s | 112 – 115 s | 113 – 118 s |
| A requests preempted | 0 | 0 | 0 | 0 | 0 – 8 | 8 – 20 |
| Output tokens/s | 276 – 296 | 446 – 477 | 611 – 660 | 711 – 802 | 711 – 802 | 711 – 801 |

- **A cap of 32 beats VTC on this trace except for A's tail.** In every
  seed A's median TTFT is 13–63% lower than VTC's and B's is 1.7–3.0 s
  lower, nothing is preempted, throughput is the same, and B's burst
  finishes between 2.1 s earlier and 1.0 s later. VTC's one win is A's p99:
  0.86–1.43 s against 4.17–6.06 s, which is 4.2–6.1× VTC's.
- **Cap 16** gives A a median close to cap 8's (0.09–0.15 s against
  0.08–0.37 s) and a lower p99 than caps 8 and 32 in every seed, and costs B
  the 28–33 s described above.
- **Caps of 8 and 4 fail B.** At 8, B's median TTFT is 42.5–47.8 s higher
  than under VTC and its burst finishes 81–93 s later. At 4 the cap is below
  A's own load (2 requests/s, each taking a median 3.4–4.7 s end to end under
  caps 8 and 16), so A queues at the gateway too: its median TTFT is
  7.2–27.5 s, and B's burst finishes 200–220 s later than under VTC.
- The best value depends on the trace: on mixed request shapes the cap of
  16 more than doubled the run time ([below](#different-request-shapes-and-what-a-token-costs)).

The first table's preemption row is VTC's cost. Each time A's queue empties
and a new request arrives, the lift puts A level with B's counter. For a moment A can
then be the most-served tenant, and after the sort vLLM's KV-pressure victim
is always the newest request of the most-served tenant. They are not what
keeps A's p99 above priority's, though: removing them with the `kv_share`
order (next section) left A's TTFT where it was.

### Choosing the preemption victim by KV share

With `preempt_order: "kv_share"` in the policy (arm `vtc-kv`), tenants
holding more than their weighted share of the running requests' KV (counted
as computed tokens) go to the back of `running`, keeping counter order inside
each group. vLLM's victim is then the newest request of the highest-counter
tenant among them, so a light tenant within its share is not preempted while
a heavier one is running. The plugin still never preempts on its own. Both
orders ran in one session, 3 seeds each (`results/kv_victim_*`):

| | Noisy neighbour, counter | Noisy neighbour, kv_share | Aggressive, counter | Aggressive, kv_share |
|---|---|---|---|---|
| A requests preempted | 7 – 15 | **0** | 16 | 11 – 16 |
| B requests preempted | 28 – 31 | 33 – 38 | 56 | 50 – 55 |
| A TTFT median | 0.41 – 0.51 s | 0.33 – 0.56 s | 16.4 s | 16.3 – 16.5 s |
| A TTFT p99 | 0.95 – 1.65 s | 0.95 – 1.73 s | 33.3 s | 32.6 – 32.8 s |
| A : B share | 0.28 – 0.31 : 0.69 – 0.72 | 0.29 : 0.71 | 0.50 : 0.50 | 0.50 : 0.50 |
| Output tokens/s | 731 – 802 | 731 – 802 | 1,304 – 1,306 | 1,310 – 1,311 |
| Recompute billed to `system` | 36.1k – 40.6k | 35.7k – 42.8k | 68.0k | 64.5k |

- In the noisy-neighbour workload the light tenant is no longer preempted,
  and B absorbs 4–8 more preemptions per seed. A's TTFT does not improve: its
  p99 is the same in two seeds and 0.08 s higher in the third.
- In the aggressive workload both tenants are backlogged peers, so A is
  sometimes over its KV share and still loses requests. It recomputes 5%
  less. Throughput was 0.3–0.5% higher per seed with `counter` always run
  first. Rerun over 6 seeds with the order alternated
  (`results/kv_victim_rotated/`, compared with `bench/paired.py`),
  `kv_share` was faster in every seed: +0.68% on average, 95% interval
  +0.13% to +1.23%. The two largest gaps, +1.58% and +1.05% in seeds 1 and
  2, came with a 30 MHz higher median SM clock and a 5–8 °C lower peak
  temperature in the `kv_share` run; in seeds 3–6, at an equal or lower
  clock, the gap is +0.29% to +0.43%. So `kv_share` is resolved as faster,
  but its size within that interval is not.
- The order also changes who gets the step's token budget first: an
  over-share tenant's running requests now come after everyone else's.

The default stays `counter`. `kv_share` moves preemptions from the light
tenant to the heavy one without a measured latency gain, and every other
result here uses the default.

### Aggressive tenant

**Workload.** Both tenants are backlogged from t = 0 with the same request
shape: 1,024-token prompts and 256 output tokens. A sends 100 requests and B
sends 300, three times the demand. The weights are equal.

| | FCFS | Priority (A high) | Priority (B high) | VTC |
|---|---|---|---|---|
| A : B share | 0.25 : 0.75 | 1.00 : 0.00 | 0.00 : 1.00 | **0.50 : 0.50** |
| A TTFT median / p99 | 32.6 / 72.0 s | 4.8 / 16.1 s | 61.0 / 72.4 s | 16.5 / 33.3 s |
| A's last request finishes | 77.9 – 78.5 s | 29.2 – 29.5 s | 77.8 – 78.6 s | 46.3 – 46.8 s |
| B's last request finishes | 77.9 – 78.5 s | 77.7 – 78.2 s | 68.6 – 69.4 s | 78.4 – 79.1 s |
| Output tokens/s (median) | 1,309 | 1,312 | 1,314 | 1,295 |

TTFT cells are medians over the 5 seeds; no cell varied by more than 0.9 s
across seeds. VTC's throughput is 0.5–1.1% below FCFS's with the same seed,
with VTC run last in every seed. With the order rotated over 6 seeds the gap
is 0.53% (95% interval 0.23–0.83%; [Scheduler overhead](#scheduler-overhead)).
- **FCFS** serves each tenant in proportion to what it sends.
- **Static priority** is all or nothing between peers.
- **VTC** splits service evenly until A runs out of work, then gives B
  everything.

The throughput cost comes with extra preemption recompute. Alternating
between two backlogged tenants churns the KV cache: VTC billed 68.0k–69.5k
weighted tokens of recompute to `system`, against 59.4k–63.8k for FCFS.

![Aggressive tenant: billed service over time](results/aggressive/service.png)

### Weighted shares

**Workload.** Each tenant has 200 identical requests backlogged from t = 0. A
has weight 2, B weight 1.

| | FCFS | VTC |
|---|---|---|
| A : B share | 0.50 : 0.50 | **0.67 : 0.33** (target 0.667) |
| A TTFT median | 32.6 – 33.4 s | 25.2 – 28.4 s |
| B TTFT median | 32.7 – 33.4 s | 56.3 – 58.0 s |
| Output tokens/s (median) | 1,304 | 1,278 (−2.0%) |

Here the cost is larger. VTC preempted 72–87 requests against FCFS's 68–70,
and recomputed 74.8k–90.3k weighted tokens against 59.4k–63.8k.

### Different request shapes, and what a token costs

**Workload.** Both tenants are backlogged from t = 0 with equal weights but
different shapes. A sends 200 prefill-heavy requests (2,048-token prompts,
32 output tokens), B 200 decode-heavy ones (128-token prompts, 512 output
tokens). 3 seeds per arm, all in one session (`results/mixed_shapes/`).

VTC's share is in its own billing units, prefill tokens at weight 1 and
decode tokens at 2. To see what the GPU actually spent, `bench/gpu_time.py`
fits each step's wall time to the work the step log records for it:

    step time = c0 + cq · (query tokens run as prefill) + cd · (decode steps) + ca · (causal attention pairs)

With async scheduling a step is logged while the one before it is still on
the GPU, so the fit pairs step k's work with the interval between the next
two log records. That alignment fits with R² = 0.997 and a median error of
0.54 ms over 29,744 steps; pairing with the same or later intervals gives
R² 0.63–0.68. Each step's time is then split between tenants in proportion
to their predicted variable cost (the fixed c0 is shared the same way).

| | FCFS | FCFS + gateway cap | VTC | VTC, kv_share |
|---|---|---|---|---|
| A : B share, billed (VTC units) | 0.72 : 0.28 | see below | **0.50 : 0.50** | 0.50 : 0.50 |
| A : B share, fitted GPU time | 0.76 : 0.24 | see below | 0.53 – 0.54 : 0.46 – 0.47 | 0.54 : 0.46 |
| A TTFT median | 21.8 – 21.9 s | 21.5 – 21.6 s | 34.1 – 34.2 s | 34.2 – 34.4 s |
| B TTFT median | 21.4 s | 86.7 – 87.1 s | 2.6 – 2.8 s | 2.8 s |
| A's last request finishes | 55.0 – 55.2 s | 43.7 – 44.0 s | 56.4 – 56.7 s | 56.6 – 56.7 s |
| B's last request finishes | 64.4 – 64.6 s | 146.1 – 146.5 s | 63.3 – 63.6 s | 63.5 – 63.6 s |
| Output tokens/s | 1,684 – 1,688 | 743 – 745 | 1,711 – 1,718 | 1,710 – 1,713 |

- **The fitted costs are not VTC's 1:2.** On this workload the fit gives
  14.6 ms per step plus 0.081 ms per prefill token, 0.079 ms per decode step
  and 8.9 ns per attention pair, so at the margin a decode token costs about
  the same as a prefill token (0.97×), not twice as much. Fitted separately,
  the `aggressive` and `noisy_neighbor` runs give 1.49× and 1.71× (R² 0.998
  and 0.977), with attention coefficients between 4.8 and 48 ns per pair, so
  the ratio depends on the batch mix and no single pair of weights is right.
- **So VTC's even split is not an even GPU split.** Its 0.50 : 0.50 in billed
  units gave the prefill-heavy tenant 53–54% of the fitted GPU time: decode
  is overpriced here, and the decode-heavy tenant gets less GPU than its
  billed half. FCFS shows the same direction (0.72 billed, 0.76 GPU time).
- VTC still does what it is for: B's median TTFT drops from 21.4 s to
  2.6–2.8 s, A's rises from 21.8 s to 34.1–34.2 s, and throughput is 1.6%
  higher than FCFS's.
- **The cap fails on this mix.** With at most 16 of B's decode-heavy
  requests in the engine, B's median TTFT is 86.7–87.1 s and the run takes
  146 s instead of 64. Most of that time B decodes alone in batches of at
  most 16, so its work costs 105 GPU-seconds against 22 under VTC. The billed
  and GPU shares over steps where both tenants wait in the engine are not
  comparable for this arm, because B's backlog sits in the gateway.

![Mixed shapes: billed service over time](results/mixed_shapes/service.png)

**Billing decode at the fitted weight.** VTC with `w_d = 0.97` (arm
`vtc-fit`) ran next to the default `w_d = 2` in one session, 3 seeds each
(`results/fit_weights_*`; GPU time from each set's own fit, R² 0.998 and
0.977):

| | Mixed shapes, w_d = 2 | Mixed shapes, w_d = 0.97 | Noisy neighbour, w_d = 2 | Noisy neighbour, w_d = 0.97 |
|---|---|---|---|---|
| A : B share, billed | 0.50 : 0.50 | 0.50 : 0.50 | 0.29 – 0.31 : 0.69 – 0.71 | 0.27 – 0.30 : 0.70 – 0.73 |
| A : B share, fitted GPU time | 0.53 – 0.54 : 0.46 – 0.47 | 0.40 : 0.60 | 0.32 – 0.34 : 0.66 – 0.68 | 0.29 – 0.31 : 0.69 – 0.71 |
| A TTFT median | 34.0 – 34.2 s | 34.4 – 34.6 s | 0.39 – 0.53 s | 0.51 – 0.56 s |
| A TTFT p99 | 54.8 – 55.1 s | 55.1 – 55.5 s | 0.95 – 1.74 s | 1.14 – 2.15 s |
| B TTFT median | 2.6 – 2.8 s | 2.6 s | 48.7 – 49.8 s | 48.6 – 50.3 s |
| B TTFT p99 | 28.3 – 28.5 s | 22.3 – 22.5 s | 97.6 – 99.2 s | 98.2 – 100.1 s |
| Output tokens/s | 1,717 – 1,723 | 1,723 – 1,733 | 731 – 802 | 731 – 802 |
| A requests preempted | 0 – 2 | 0 – 1 | 8 – 25 | 21 – 23 |

- The GPU-time split does not land on 0.50 : 0.50; it overshoots to
  0.40 : 0.60. The fitted weight is a marginal cost and leaves out the
  12.7–14.6 ms that every step costs. In seed 1 the backlogged steps carried
  88.7k of A's prefill tokens instead of 147.5k, with about the same number
  of B's decode steps (64.5k against 63.6k), and each step's fixed cost is
  attributed in proportion to its variable cost, so more of it lands on B.
  The weight that evens GPU time here lies somewhere between 0.97 and 2; it
  was not searched for. This session's refit also gave a ratio of 1.08, not
  0.97.
- On the mixed workload B gains (p99 TTFT 22.3–22.5 s instead of
  28.3–28.5 s, throughput 0.0–0.9% higher per seed) and A's median TTFT rises
  by 0.2–0.5 s.
- On the noisy neighbour the interactive tenant gets slower: median TTFT
  0.51–0.56 s and p99 1.14–2.15 s, with 21–23 preemptions. B's TTFT and the
  throughput do not change.

The default stays `w_d = 2`: a single fitted ratio did not equalize GPU time,
and it slowed the interactive tenant.

### Long prompts under KV pressure

**Workload.** Long prompts of the length that livelocked the earlier design
(on a different model, budget and KV size):
- Tenant B sends 6 requests with 6,144-token prompts at t = 0.
- Tenant A sends Poisson 1 req/s short requests.
- The KV cache is capped at 520 blocks (8,320 tokens), so one long request
  nearly fills it.

| | FCFS | VTC |
|---|---|---|
| B's six requests finish | 16.1 – 16.3 s | 16.1 – 16.3 s |
| A TTFT p99 | 10.3 – 12.8 s | **1.9 – 3.0 s** |
| Preemptions (all of tenant A) | 3 – 6 | **0** |
| Fewest free KV blocks during the run | 0 | 60 – 76 |

Under FCFS, A's short requests wait behind B's long prompts: over the steps
in which both tenants had requests waiting, A received none of the billed
service (backlogged share 0.0 in every seed). When the cache ran out, vLLM
also preempted 3–6 of A's requests, the newest running ones. The summaries do
not separate how much each of the two adds to A's p99. VTC admits A ahead of
B's queued prompts (backlogged share 0.09–0.14 for A), the cache never runs
out, and B's long requests finish at the same time. The earlier
quantum-preemption design was not run on this configuration.

### Scheduler overhead

`bench/hook_overhead.py` measures CPU time per `schedule()` +
`update_from_output()` call in the real vLLM scheduler, with a stubbed model
runner. `scripts/hook-overhead.sh` ran it five times, each run in a fresh
container with the three variants measured in a rotated order
(`results/hook_overhead.jsonl`; every row records the commit and tree
hashes, and [results/README.md](results/README.md#run-history) lists the
earlier single run it replaced). Medians over the five runs, with the range of the per-run
differences against native:

| Running requests | vLLM native | Plugin, passthrough | Plugin, VTC | Added by passthrough | Added by VTC |
|---|---|---|---|---|---|
| 64 | 428 µs | 585 µs | 599 µs | 170 µs (92 – 182) | 172 µs (165 – 379) |
| 128 | 825 µs | 1,112 µs | 1,135 µs | 287 µs (269 – 321) | 311 µs (289 – 348) |
| 256 | 1,689 µs | 2,217 µs | 2,389 µs | 572 µs (350 – 619) | 674 µs (388 – 1,171) |

- Most of the added time is in the hooks, which passthrough runs too:
  sorting `running`, billing and the per-step log. VTC's decisions add
  2–102 µs more at the median, inside the run-to-run range of the
  passthrough cost at every size.
- The host was shared with other work (1-minute load average 8.3–9.2 on
  32 CPUs), and single runs differ by up to 3× (VTC at 256 requests:
  388–1,171 µs added).
- At 128 running requests a Qwen2.5-7B decode step takes about 50 ms
  (median TPOT in these runs), so the hooks add about 0.6%.
- With async scheduling this CPU time overlaps the previous step's GPU work.

End to end, the overhead has to be measured on a saturating trace. On the
open-loop noisy-neighbour trace the arrival schedule sets how long a run
lasts, so every arm's seed-1 run gives 730.6–731.2 output tokens/s, with or
without the plugin. The `aggressive` workload saturates the engine instead:
all 400 requests arrive in the first 0.3 s, so throughput is the engine's
capacity. It was run with vLLM without the plugin, the plugin in
passthrough and VTC, six seeds in one session, with the order of the three
arms rotated so that each arm ran first, second and third in two seeds each
(`results/overhead_rotated/`). `bench/paired.py` compares the arms seed for
seed and gives a 95% t interval for the mean difference
(`results/overhead_rotated/paired.json`); a gap whose interval includes zero
is called unresolved.

| | vLLM, no plugin | Plugin, passthrough | Plugin, VTC |
|---|---|---|---|
| Output tokens/s | 1,313 – 1,334 (median 1,318) | 1,313 – 1,335 (median 1,317) | 1,306 – 1,321 (median 1,311) |
| Per seed, against the arm to the left | | −0.11% to +0.04% | −1.03% to −0.18% |
| Mean (95% interval) | | −0.035% (−0.10% to +0.03%) | −0.53% (−0.83% to −0.23%) |
| TPOT median | 48.6 – 49.4 ms | 48.6 – 49.4 ms | 49.1 – 50.1 ms |
| TTFT median, A / B | 32.0 – 32.6 / 32.2 – 32.8 s | 31.9 – 32.8 / 32.1 – 32.8 s | 16.2 – 16.6 / 44.5 – 45.0 s |

- **Passthrough against no plugin is unresolved.** The interval spans zero:
  if the hooks cost throughput end to end, it is less than about 0.1% here.
- **VTC against passthrough is resolved:** 0.53% lower, slower in all six
  seeds (0.56% below no plugin, interval 0.29–0.83%).
- Run order does not account for either: averaged over seeds, a seed's
  first, second and third runs are +0.05%, −0.10% and +0.04% from that
  seed's mean.
- Each arm's throughput spreads 1.1–1.7% across seeds, more than these gaps.
  That spread moves with the GPU's state, which cannot be locked: the median
  SM clock was 2,610–2,640 MHz in seeds 1–2 and 2,565–2,595 MHz in seeds 3–6,
  with peaks of 75–83 °C. It acts on all arms of a seed alike, which is why
  the comparison is paired.
- An earlier set with the arms in a fixed order (no plugin, passthrough,
  VTC; three seeds, `results/overhead/`) had passthrough 0.13–0.50% below
  no plugin. That gap did not reproduce with the order rotated. Its VTC gap,
  0.67–0.74% below passthrough, is inside the interval above.
- VTC's gap is not all overhead in any case: VTC also reorders the
  work (A's TTFT halves, B's rises) and preempts more, recomputing 67.7k–68.0k
  weighted tokens against 63.8k for passthrough.
- Throughput also differs between sessions for the same arm (the fixed-order
  set's no-plugin median is 1.0% above this set's), so compare arms within
  one set.

## Reproduce

Requirements: Linux with Docker and the NVIDIA Container Toolkit, one 24 GB
GPU (the results are from an RTX 4090), `nvidia-smi`, and Python 3.12 or
later on the host. The runner itself uses only the standard library; the
engine runs in the vLLM image pinned by digest in `configs/pins.json`, and Go
code builds in the pinned Go image.

1. **Models.** `scripts/fetch-models.sh` (or `make models`) downloads the
   pinned revision of Qwen2.5-7B-Instruct (about 15 GB) into
   `.cache/huggingface`. The engine container mounts that directory and runs
   with `HF_HUB_OFFLINE=1`, so the download must happen first.
2. **Binaries.** `scripts/build-go.sh` (or `make build`) builds
   `bin/fairsched-proxy` and `bin/fairsched-loadgen`, which the runner starts
   for every run.
3. **Plots.** `scripts/run-all.sh` draws the figures with `.venv/bin/python`:
   `python3 -m venv .venv && .venv/bin/pip install matplotlib==3.11.2`
   (`make .venv/bin/python`).
4. **Runs.** `scripts/run-all.sh` (or `make bench`) runs every experiment
   above, about 5 h. A single run is
   `python3 -m bench.run noisy_neighbor vtc --seed 1`. Each run waits for an
   exclusive `flock` on `/var/tmp/rtx4090.lock` (`FAIRSCHED_GPU_LOCK`), refuses
   to start while another process holds more than 1 GiB of GPU memory, starts
   its own engine with `--gpu-memory-utilization 0.83`, and writes raw output
   to `results/raw/<workload>/<arm>-s<seed>/`. It never overwrites an existing
   run directory.
5. **Summaries.** `python3 -m bench.report <workload> ... [--plots]` rewrites
   `results/<workload>/summary.json` (and the PNGs) from the raw runs, and
   `python3 -m bench.gpu_time <workload>` writes the step-time fit and GPU-time
   shares to `results/<workload>/gpu_time.json`.
   `python3 -m bench.paired <name> <arm>:<arm> ...` compares arms seed for
   seed on throughput and writes `results/<name>/paired.json`.

## Testing

`scripts/test.sh` (or `make test`) runs three suites, in this order. The
first two need no model; the third needs the test model's `config.json`,
which `scripts/fetch-models.sh --test-only` (`make test-model`) fetches.
`make test-local` runs the first two without Docker, using a local Go
toolchain.

CI (`.github/workflows/ci.yml`) runs only the first two suites. The real-vLLM
scheduler tests, the only ones that exercise the plugin inside vLLM, skip
there because vLLM is not installed, so a change that
breaks the plugin's vLLM integration passes CI. Run `scripts/test.sh` before
relying on such a change.

- **Policy, report, step log, runners, step-time fit and paired comparison**
  (`tests/test_policy.py`, `tests/test_report.py`, `tests/test_audit.py`,
  `tests/test_bench.py`, `tests/test_gpu_time.py`, `tests/test_paired.py`,
  pure Python): the lift rule
  (waiting queue only), weighted shares, recompute and waste attribution,
  unlisted tenants, queue semantics matching vLLM's `FCFSRequestQueue`, the
  report's percentiles, shares and timelines, dropped-record reporting, the
  runner's gateway and image settings, the replica runner's routing and
  counter settings, sticky session placement, the cross-replica share, and
  the paired differences' intervals and run positions.
- **Shared counters** (`tests/test_counters.py`, pure Python): the counter
  service's replace-with-cumulative semantics and input checks; the engine-side
  sync against a live service (each new remote total handed over once, a
  periodic thread, a timeout on a hung service); the fallback (an unreachable
  service is logged once, local counting goes on with the other replicas'
  totals held, a restarted service is whole again after one round, a remote
  total never decreases); remote service in selection, lift and victim order,
  lifts never published; and a two-replica simulation in which shared
  counters move A's share from 0.25 to 0.50.
- **Go** (`go test ./...`): the gateway's header stripping and tenant
  injection, streaming, cancellation, round-robin and sticky routing over
  several upstreams, and the load generator (including its session header).
- **Real vLLM scheduler on CPU** (`tests/test_scheduler_vllm.py`), in the
  pinned container with no GPU. It builds the real vLLM 0.31.0
  `AsyncScheduler`, KV cache manager, block pool and prefix cache, replaces
  only the model forward pass, and checks:
  - a tenant arriving behind a 16k-token backlog (eight 2,048-token prompts)
    gets its first token within 6 steps; under FCFS it takes at least 25;
  - under KV pressure only the over-served tenant is preempted, where native
    FCFS preempts the newly arrived tenant;
  - with three 6,144-token prompts and ten short requests in a small KV
    cache, every request finishes and none is preempted more than twice;
  - with the `kv_share` order, a light tenant that the default order
    preempts after a lift is not preempted;
  - recompute is billed to `system`;
  - aborting a request with a step in flight marks exactly that step's tokens
    as wasted;
  - prefix-cache hits are not billed;
  - the step log records each tenant's prefill, decode and attention work;
  - 2:1 weights give a 2:1 split;
  - with two schedulers sharing a live counter service, service on one
    replica reorders admission on the other, and with the service down the
    engine logs it and keeps the single-engine order; with no counter URL
    nothing is shared, and passthrough mode refuses one.

## Repository layout

| Path | What it is |
|---|---|
| `fairsched/scheduler.py` | the vLLM scheduler plugin (the three hooks) |
| `fairsched/policy.py` | VTC counters, lift rule, cost model, accounts |
| `fairsched/queue.py` | per-tenant waiting queue behind vLLM's `RequestQueue` interface |
| `fairsched/audit.py` | per-step JSONL log |
| `fairsched/counters.py` | counters shared across replicas: the service and the engine-side sync (off by default) |
| `proxy/`, `cmd/fairsched-proxy` | Go gateway: API key → tenant, trusted `vllm_xargs`, streaming, cancellation |
| `loadgen/`, `cmd/fairsched-loadgen` | Go open-arrival load generator with per-chunk timestamps |
| `bench/` | workloads, experiment runner, report, plots, step-time fit, paired comparison, hook overhead; `replicas.py` and `replica_report.py` for runs on several engines |
| `docs/` | [design](docs/design.md), [design trade-offs](docs/design-tradeoffs.md), [livelock post-mortem](docs/livelock.md) |

## Limitations

- The plugin hooks vLLM internals, so it is pinned to vLLM 0.31.0. The CPU
  scheduler tests are the upgrade check.
- Single engine and GPU. Counters live in memory and reset when the engine
  restarts.
- It refuses to run with speculative decoding, LoRA or KV connectors.
- The cost model is VTC's linear one, with weights 1 for prefill and 2 for
  decode. It does not model attention cost growing with context length.
  Fitted step times put a decode token at 0.97–1.71× a prefill token
  depending on the workload, so with mixed shapes an even billed split gave
  the prefill-heavy tenant 53–54% of GPU time; billing decode at the fitted
  0.97 overshot to 40%. Attention (context length) is not billed. "Share" in
  the tables is in billing units unless it says GPU time.
- The evaluation is one RTX 4090, one model (Qwen2.5-7B) and three to six
  seeds per arm, on synthetic traces in which all of a tenant's requests have the same
  shape except in `mixed_shapes`. The gateway cap was swept (4, 8, 16, 32)
  only on the noisy-neighbour trace; in `mixed_shapes` it ran at 16.
- Fairness acts on admission and token-budget order. A request already
  running a very long prefill keeps its chunks going. VTC accepts that
  bound in exchange for never throwing away work.
- Under KV pressure the light tenant can be the victim with the default
  victim order, as the noisy-neighbour results show. The `kv_share` order
  avoids that by preempting the heavy tenant more, and did not improve the
  light tenant's latency.
