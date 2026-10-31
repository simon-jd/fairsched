#!/usr/bin/env bash
# Download the pinned model revisions (configs/pins.json) into .cache/huggingface,
# the cache that the benchmark and test containers mount read-only with
# HF_HUB_OFFLINE=1. Uses huggingface_hub's CLI from the pinned vLLM image, run
# as the current user. Set HF_TOKEN if your network needs authenticated access.
#   scripts/fetch-models.sh              Qwen2.5-7B-Instruct (about 15 GB) and the test model's config
#   scripts/fetch-models.sh --test-only  only Qwen2.5-1.5B-Instruct's config.json (enough for scripts/test.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
hf=${FAIRSCHED_HF_CACHE:-$PWD/.cache/huggingface}
mkdir -p "$hf"
image=$(python3 -c 'import json; p = json.load(open("configs/pins.json")); print(p["image"].split(":")[0] + "@" + p["image_digest"])')

revision() {
  python3 -c 'import json, sys; print(json.load(open("configs/pins.json"))["models"][sys.argv[1]])' "$1"
}

fetch() {
  docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf \
    ${HF_TOKEN:+-e HF_TOKEN} ${HF_HUB_OFFLINE:+-e HF_HUB_OFFLINE} \
    -v "$hf:/hf" --entrypoint hf "$image" download "$@"
}

fetch Qwen/Qwen2.5-1.5B-Instruct config.json --revision "$(revision Qwen/Qwen2.5-1.5B-Instruct)"
if [[ "${1:-}" != "--test-only" ]]; then
  fetch Qwen/Qwen2.5-7B-Instruct --revision "$(revision Qwen/Qwen2.5-7B-Instruct)"
fi
