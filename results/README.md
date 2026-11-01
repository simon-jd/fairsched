# Results

- `<workload>/summary.json`: one entry per (arm, seed) run, written by
  `bench/report.py` from the raw run directories under `results/raw/`, which
  are not committed. The README's tables and plots come from these files.
- `<workload>/*.png`: plots of seed 1 (`python3 -m bench.report ... --plots`).
- `noisy_neighbor/` and `aggressive/` hold seeds 1–5 of their original arms;
  seeds 4 and 5 were run in a later session than seeds 1–3, with the code of
  that time (their `source_tree` says which).
- `overhead/summary.json`: the `aggressive` workload rerun with vLLM without
  the plugin (`native`), the plugin in passthrough (`fcfs`) and VTC, three
  seeds each, interleaved by seed in one session
  (`python3 -m bench.run aggressive native fcfs vtc --name overhead ...`).
  These runs are separate from `aggressive/summary.json`.
- `kv_victim_noisy_neighbor/` and `kv_victim_aggressive/`: VTC with the
  default preemption order (`vtc`) and with `preempt_order: "kv_share"`
  (`vtc-kv`), three seeds each, interleaved in one session.
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
- `hook_overhead.jsonl`: the CPU microbenchmark from `bench/hook_overhead.py`
  (a single run). It records no source revision.

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
