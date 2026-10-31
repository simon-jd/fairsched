"""CPU cost of the plugin's hooks per scheduling step, in the real vLLM scheduler.

    docker run ... vllm/vllm-openai:v0.31.0 -m bench.hook_overhead   (see scripts/test.sh for mounts)

Builds native AsyncScheduler, the plugin in passthrough mode and the plugin in
VTC mode with the same KV cache and requests: ``batch`` decoding requests
split across two tenants plus a queue of waiting ones. Times schedule() +
update_from_output() per step with a fake model runner, so the number is
scheduler CPU time only. A GPU step for Qwen2.5-7B at this batch size takes
tens of milliseconds.
"""
from __future__ import annotations

import json
import statistics
import time

from tests.vllm_harness import Engine, make_request, make_scheduler


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


def main() -> None:
    rows = []
    for batch in (64, 128, 256):
        native = per_step_us(native=True, mode="passthrough", batch=batch)
        passthrough = per_step_us(native=False, mode="passthrough", batch=batch)
        vtc = per_step_us(native=False, mode="vtc", batch=batch)
        rows.append({"running_requests": batch, "native_us": round(native), "passthrough_us": round(passthrough),
                     "vtc_us": round(vtc), "vtc_overhead_us": round(vtc - native)})
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
