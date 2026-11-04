# Results

## Run history

Each set of runs, in the order it was committed. Commit and date are when
the file, or that set of runs inside it, was committed to this repository.
The runs in `fe04a1f` were made before this history starts, at an earlier
revision of the code (see [`source_tree`](#source_tree)).

| File | Commit | Date | What it measured | Seeds |
|---|---|---|---|---|
| `noisy_neighbor/summary.json` | `fe04a1f` | 2026-10-30 | Noisy neighbour: FCFS, priority (A high), VTC; vLLM without the plugin (`native`) on one seed | 1–3 (`native`: 1) |
| `aggressive/summary.json` | `fe04a1f` | 2026-10-30 | Aggressive tenant: FCFS, priority with A high and with B high, VTC | 1–3 |
| `weighted/summary.json` | `fe04a1f` | 2026-10-30 | Weighted shares (A 2, B 1): FCFS, VTC | 1–3 |
| `long_prompt/summary.json` | `fe04a1f` | 2026-10-30 | Long prompts under KV pressure: FCFS, VTC | 1–3 |
| `hook_overhead.jsonl`, first version | `fe04a1f` | 2026-10-30 | Hook CPU time, one run, no source revision recorded. It reported only VTC minus native (132–403 µs) as the added cost; passthrough minus native was 126–469 µs, so it understated the hook cost at 256 requests. Replaced in `f9dc819` | 1 run |
| `noisy_neighbor/summary.json` | `4e2c702` | 2026-10-31 | Noisy neighbour with the gateway cap at 16 (`fcfs-cap`) | 1–3 |
| `overhead/summary.json` | `f60b1b1` | 2026-10-31 | `aggressive` with no plugin, passthrough and VTC in a fixed order | 1–3 |
| `noisy_neighbor/summary.json`, `aggressive/summary.json` | `0105127` | 2026-10-31 | Seeds 4 and 5 of each file's first arms (not `fcfs-cap` or `native`), run in a later session than seeds 1–3 with the code of that time (their `source_tree` says which) | 4–5 |
| `kv_victim_noisy_neighbor/`, `kv_victim_aggressive/` `summary.json` | `a371fda` | 2026-10-31 | VTC with the default and the `kv_share` victim order, interleaved | 1–3 |
| `mixed_shapes/`, `kv_victim_*/gpu_time.json` | `6ab7b69` | 2026-10-31 | Prefill-heavy and decode-heavy tenants: FCFS, gateway cap 16, VTC, VTC with `kv_share`; the step-time fit | 1–3 |
| `fit_weights_mixed_shapes/`, `fit_weights_noisy_neighbor/` | `9c623eb` | 2026-10-31 | VTC with `w_d = 2` and with the fitted `w_d = 0.97` | 1–3 |
| `hook_overhead.jsonl` | `f9dc819` | 2026-11-02 | Hook CPU time, five runs at commit `e6f4f47`, variant order rotated | 5 runs |
| `livelock/summary.json` | `c8bb1a2` | 2026-11-02 | Counts from the earlier design's step log; the log and that design's code are not committed | 1 run |
| `overhead_rotated/`, `kv_victim_rotated/` | `2d7a9c2` | 2026-11-02 | `aggressive` with the arm order rotated: no plugin, passthrough and VTC; VTC and VTC with `kv_share`; `paired.json` | 1–6 |
| `cap_sweep/summary.json` | `0b20d12` | 2026-11-02 | Noisy neighbour with the gateway cap at 4, 8, 16 and 32, arm order rotated | 1–5 |

## Files

- `<workload>/summary.json`: one entry per (arm, seed) run, written by
  `bench/report.py` from the raw run directories under `results/raw/`, which
  are not committed. The README's tables and plots come from these files.
- `<workload>/*.png`: plots of seed 1 (`python3 -m bench.report ... --plots`).
- `noisy_neighbor/` and `aggressive/` hold seeds 1–5 of their original arms.
- `overhead/summary.json`: the `aggressive` workload rerun with vLLM without
  the plugin (`native`), the plugin in passthrough (`fcfs`) and VTC, three
  seeds each, interleaved by seed in one session
  (`python3 -m bench.run aggressive native fcfs vtc --name overhead ...`).
  These runs are separate from `aggressive/summary.json`.
- `kv_victim_noisy_neighbor/` and `kv_victim_aggressive/`: VTC with the
  default preemption order (`vtc`) and with `preempt_order: "kv_share"`
  (`vtc-kv`), three seeds each, interleaved in one session.
- `overhead_rotated/` and `kv_victim_rotated/`: the same two comparisons on
  the `aggressive` workload repeated with six seeds and the arm order rotated
  per seed (each of `native`, `fcfs`, `vtc` first, second and third in two
  seeds; `vtc` and `vtc-kv` each first in three), the two sets interleaved
  by seed in one session (`python3 -m bench.run aggressive native fcfs vtc
  --name overhead_rotated --seed 1`, with the arms in the seed's order).
  Each `paired.json` is the seed-for-seed throughput comparison from
  `bench/paired.py`
  (`python3 -m bench.paired overhead_rotated fcfs:native vtc:fcfs vtc:native`,
  `python3 -m bench.paired kv_victim_rotated vtc-kv:vtc`): per-seed
  differences, their mean and 95% t interval, each run's position in its
  seed, the mean effect of position, and the GPU clock and temperature.
- `mixed_shapes/`: a prefill-heavy and a decode-heavy tenant under FCFS, the
  gateway cap, VTC and VTC with `kv_share`, three seeds each, in one session.
- `fit_weights_mixed_shapes/` and `fit_weights_noisy_neighbor/`: VTC with the
  default `w_d = 2` (`vtc`) and with the fitted `w_d = 0.97` (`vtc-fit`),
  three seeds each, interleaved in one session.
- `*/gpu_time.json` (`mixed_shapes`, `kv_victim_*`, `fit_weights_*`): the step-time fit from
  `bench/gpu_time.py` and each run's GPU-time and billed shares. The fit is
  over all runs of the set.
- `noisy_neighbor/summary.json` also holds the `fcfs-cap` arm (FCFS with the
  gateway capping each tenant at 16 requests in flight), run in the same
  later session as `overhead/`.
- `cap_sweep/summary.json`: the `noisy_neighbor` workload with the gateway
  cap at 4, 8, 16 and 32 requests per tenant (`fcfs-cap4`, `fcfs-cap8`,
  `fcfs-cap`, `fcfs-cap32`), seeds 1–5, in one later session with the arm
  order rotated per seed. Its FCFS and VTC baselines are the
  `noisy_neighbor/` runs with the same seeds.
- `livelock/summary.json`: counts and hashes from the step log of the earlier
  quantum-preemption design's livelocked run ([docs/livelock.md](../docs/livelock.md)),
  written by `scripts/livelock_summary.py`. The log and that design's code are
  not in this repository; the file records their SHA-256.
- `hook_overhead.jsonl`: the CPU microbenchmark from `bench/hook_overhead.py`,
  five runs from `scripts/hook-overhead.sh`, one row per run and batch size,
  each with the commit, tree hashes, the order the variants ran in and the
  host's load average.

## `source_tree`

Each run in `summary.json` records `source_tree`: the git tree hash
(`git rev-parse HEAD:<dir>`) of each source directory as published here,
namely `fairsched/` (the scheduler plugin), `bench/` (workloads, runner and
report), `cmd/`, `proxy/` and `loadgen/` (gateway and load generator) and
`configs/` (pinned image and model revisions). Compare them with
`git rev-parse HEAD:<dir>` in a checkout to see whether a directory has
changed since.

The runs recorded the commit they were started from, an earlier revision of
this code. These trees differ from that revision only in the following, none
of which changes the behaviour those runs measured:

- identifiers renamed when the project got its name: the Python package, Go
  module and commands, environment variables, `vllm_xargs` keys, HTTP headers,
  the gateway's error type and the engine container name;
- comments, help text and documentation in these directories, the
  workloads' description strings, the example gateway configuration (the
  runner writes its own), and load-generator flag defaults that the runner
  overrides;
- for `noisy_neighbor`, `aggressive` and `weighted`, the KV block override
  that only `long_prompt` uses (512 blocks then, 520 now). The `long_prompt`
  runs used 520 blocks.

Runs made since then record `source_tree` themselves: `bench/run.py` writes
the tree hashes at `HEAD` into each run's `meta.json`, together with
`source_dirty`, which is true if any of those directories had uncommitted
changes. `bench/report.py` copies both into `summary.json` (`source_dirty`
only when true). For a run whose `meta.json` predates this, it keeps the
`source_tree` already recorded for that arm and seed in `summary.json`.
