#!/usr/bin/env bash
# Every experiment in the README, one after another, three seeds each (five for
# the headline arms). Each run waits for the machine-wide GPU lock, starts its
# own engine and tears it down.
# About 4.5 h on one RTX 4090.
set -euo pipefail
cd "$(dirname "$0")/.."
seeds=(--seed 1 --seed 2 --seed 3)
python3 -m bench.run noisy_neighbor fcfs priority vtc "${seeds[@]}" --seed 4 --seed 5
python3 -m bench.run noisy_neighbor native          # no-plugin reference, one seed
python3 -m bench.run noisy_neighbor fcfs-cap "${seeds[@]}"
python3 -m bench.run aggressive native fcfs vtc --name overhead "${seeds[@]}"
python3 -m bench.run noisy_neighbor vtc vtc-kv --name kv_victim_noisy_neighbor "${seeds[@]}"
python3 -m bench.run aggressive vtc vtc-kv --name kv_victim_aggressive "${seeds[@]}"
python3 -m bench.run aggressive fcfs priority priority-b vtc "${seeds[@]}" --seed 4 --seed 5
python3 -m bench.run weighted fcfs vtc "${seeds[@]}"
python3 -m bench.run mixed_shapes fcfs fcfs-cap vtc vtc-kv "${seeds[@]}"
python3 -m bench.run long_prompt fcfs vtc "${seeds[@]}"
.venv/bin/python -m bench.report noisy_neighbor aggressive weighted long_prompt mixed_shapes --plots
python3 -m bench.report overhead kv_victim_noisy_neighbor kv_victim_aggressive
python3 -m bench.gpu_time mixed_shapes kv_victim_noisy_neighbor kv_victim_aggressive
