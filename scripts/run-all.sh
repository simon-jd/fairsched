#!/usr/bin/env bash
# Every experiment in the README, one after another, three seeds each. Each run
# waits for the machine-wide GPU lock, starts its own engine and tears it down.
# About 2.5 h on one RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."
seeds=(--seed 1 --seed 2 --seed 3)
python3 -m bench.run noisy_neighbor fcfs priority vtc "${seeds[@]}"
python3 -m bench.run noisy_neighbor native          # plugin overhead reference
python3 -m bench.run aggressive fcfs priority priority-b vtc "${seeds[@]}"
python3 -m bench.run weighted fcfs vtc "${seeds[@]}"
python3 -m bench.run long_prompt fcfs vtc "${seeds[@]}"
.venv/bin/python -m bench.report noisy_neighbor aggressive weighted long_prompt --plots
