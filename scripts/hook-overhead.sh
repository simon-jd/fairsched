#!/usr/bin/env bash
# CPU cost of the plugin's hooks (bench/hook_overhead.py), repeated in fresh
# containers of the pinned vLLM image (CPU only, no network). Every row
# records the commit, the tree hashes of the directories the benchmark runs
# and whether they had uncommitted changes.
#   scripts/hook-overhead.sh [runs] > results/hook_overhead.jsonl   (default 5 runs)
# Needs the test model's config.json (scripts/fetch-models.sh --test-only).
set -euo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
runs=${1:-5}

dirty=false
if [[ -n "$(git status --porcelain -- fairsched bench tests)" ]]; then dirty=true; fi
source_json=$(printf '{"git_commit": "%s", "source_tree": {"fairsched": "%s", "bench": "%s", "tests": "%s"}, "source_dirty": %s}' \
  "$(git rev-parse HEAD)" "$(git rev-parse HEAD:fairsched)" "$(git rev-parse HEAD:bench)" \
  "$(git rev-parse HEAD:tests)" "$dirty")

hf=${FAIRSCHED_HF_CACHE:-$root/.cache/huggingface}
snapshot=$(cd "$hf" 2>/dev/null && ls -d hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/* 2>/dev/null | head -1 || true)
if [[ -z "$snapshot" ]]; then
  echo "No Qwen2.5-1.5B-Instruct snapshot under $hf; run scripts/fetch-models.sh --test-only first." >&2
  exit 1
fi
image=$(python3 -c 'import json; p = json.load(open("configs/pins.json")); print(p["image"].split(":")[0] + "@" + p["image_digest"])')

for run in $(seq 1 "$runs"); do
  docker run --rm --network none --entrypoint python3 \
    -e HF_HUB_OFFLINE=1 -e PYTHONPATH=/workspace -e PYTHONDONTWRITEBYTECODE=1 \
    -e FAIRSCHED_TEST_MODEL="/root/.cache/huggingface/$snapshot" \
    -v "$root:/workspace:ro" -v "$hf:/root/.cache/huggingface:ro" -w /workspace \
    "$image" -m bench.hook_overhead --run "$run" --source "$source_json"
done
