# Results

- `<workload>/summary.json`: one entry per (arm, seed) run, written by
  `bench/report.py` from the raw run directories under `results/raw/`, which
  are not committed. The README's tables and plots come from these files.
- `<workload>/*.png`: plots of seed 1 (`python3 -m bench.report ... --plots`).
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
