"""CPU cost of the plugin's hooks per scheduling step, in the real vLLM scheduler.

    scripts/hook-overhead.sh 5 > results/hook_overhead.jsonl

runs this module in the pinned vLLM container, once per run in a fresh
container, and passes the source revision in with ``--source``.

Builds native AsyncScheduler, the plugin in passthrough mode and the plugin in
VTC mode with the same KV cache and requests: ``batch`` decoding requests
split across two tenants plus a queue of waiting ones. Times schedule() +
update_from_output() per step with a fake model runner, so the number is
scheduler CPU time only. A GPU step for Qwen2.5-7B at this batch size takes
tens of milliseconds. The three variants are measured in an order that
rotates with ``--run``, so no variant always goes first.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

from tests.vllm_harness import Engine, make_request, make_scheduler

VARIANTS = {"native": (True, "passthrough"), "passthrough": (False, "passthrough"), "vtc": (False, "vtc")}


def per_step_us(*, native: bool, mode: str, batch: int, steps: int = 400) -> float:
    engine = Engine(make_scheduler(native=native, mode=mode, budget=8192, num_blocks=20000, max_seqs=batch))
    for i in range(batch + 64):
        engine.add(make_request("A" if i % 4 == 0 else "B", 64, max_tokens=10_000))
    for _ in range(20):                      # admit and finish prefills
        engine.run_step()
    samples = []
    for _ in range(steps):
        start = time.perf_counter_ns()
        engine.run_step()
        samples.append((time.perf_counter_ns() - start) / 1e3)
    return statistics.median(samples)


def variant_order(run: int) -> list[str]:
    names = list(VARIANTS)
    shift = (run - 1) % len(names)
    return names[shift:] + names[:shift]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=int, default=1, help="run number; also sets the measurement order")
    parser.add_argument("--source", default="{}", help="JSON object copied into every row (source revision)")
    args = parser.parse_args()
    source = json.loads(args.source)
    with open("/proc/loadavg") as f:
        loadavg = float(f.read().split()[0])
    order = variant_order(args.run)
    for batch in (64, 128, 256):
        us = {name: per_step_us(native=VARIANTS[name][0], mode=VARIANTS[name][1], batch=batch) for name in order}
        row = {"run": args.run, "order": order, "running_requests": batch,
               "native_us": round(us["native"]), "passthrough_us": round(us["passthrough"]),
               "vtc_us": round(us["vtc"]),
               "passthrough_overhead_us": round(us["passthrough"] - us["native"]),
               "vtc_overhead_us": round(us["vtc"] - us["native"]),
               "loadavg_1m": loadavg, **source}
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
