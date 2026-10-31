#!/usr/bin/env bash
# Pure-Python tests on the host, Go tests in the golang container, then the
# real-vLLM scheduler tests in the pinned vLLM container (CPU only, no network).
# Only the last suite needs the model cache (see scripts/fetch-models.sh).
#   scripts/test.sh            all three suites
#   scripts/test.sh --no-go    skip the Go tests
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)

python3 -m unittest tests.test_policy tests.test_report tests.test_audit

if [[ "${1:-}" != "--no-go" ]]; then
  go_image=$(python3 -c 'import json; print(json.load(open("configs/pins.json"))["go_image"])')
  docker run --rm --network none -v "$root:/src:ro" -w /src -e GOFLAGS=-buildvcs=false -e GOCACHE=/tmp/gocache \
    "$go_image" go test ./proxy/... ./loadgen/... ./cmd/...
fi

hf=${FAIRSCHED_HF_CACHE:-$root/.cache/huggingface}
model=Qwen/Qwen2.5-1.5B-Instruct
snapshot=$(cd "$hf" 2>/dev/null && ls -d hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/* 2>/dev/null | head -1 || true)
if [[ -z "$snapshot" ]]; then
  echo "No $model snapshot under $hf; run scripts/fetch-models.sh --test-only first." >&2
  exit 1
fi
image=$(python3 -c 'import json; p = json.load(open("configs/pins.json")); print(p["image"].split(":")[0] + "@" + p["image_digest"])')

# Only the model's config.json is read; no weights are loaded.
docker run --rm --network none --entrypoint python3 \
  -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
  -e FAIRSCHED_TEST_MODEL="/root/.cache/huggingface/$snapshot" \
  -v "$root:/workspace:ro" -v "$hf:/root/.cache/huggingface:ro" -w /workspace \
  "$image" -m unittest tests.test_scheduler_vllm -v
