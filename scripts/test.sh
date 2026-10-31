#!/usr/bin/env bash
# Pure-Python tests on the host, then the real-vLLM scheduler tests in the pinned
# container (CPU only, no network). Go tests run in the golang container.
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
hf=${FAIRSCHED_HF_CACHE:-$root/.cache/huggingface}
snapshot=$(cd "$hf" && ls -d hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/* | head -1)

python3 -m unittest tests.test_policy tests.test_report

# Only the model's config.json is read; no weights are loaded.
docker run --rm --network none --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e FAIRSCHED_TEST_MODEL="/root/.cache/huggingface/$snapshot" \
  -v "$root:/workspace:ro" -v "$hf:/root/.cache/huggingface:ro" -w /workspace \
  vllm/vllm-openai:v0.31.0 -m unittest tests.test_scheduler_vllm -v

if [[ "${1:-}" != "--no-go" ]]; then
  docker run --rm --network none -v "$root:/src:ro" -w /src -e GOFLAGS=-buildvcs=false -e GOCACHE=/tmp/gocache \
    golang:1.26.8-bookworm go test ./proxy/... ./loadgen/... ./cmd/...
fi
