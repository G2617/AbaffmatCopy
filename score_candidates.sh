#!/bin/bash
# Copyright 2026 Eamon Byrne
# Licensed under the Apache License, Version 2.0; see LICENSE.


export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
export TRIANGLE_MULTIPLICATIVE=torch

INPUT_CSV_PATH="${1:-./data/Covid/Covid_design.csv}"
SCORE_INPUT_JSON_PATH="${2:-$(dirname "$INPUT_CSV_PATH")/$(basename "$INPUT_CSV_PATH" .csv)}"

python prepare_scoring_inputs.py \
  --input_csv_path "$INPUT_CSV_PATH" \
  --output_dir "$SCORE_INPUT_JSON_PATH"

torchrun --standalone --nproc_per_node=1 runners/score_complex.py --project "Covid-design" --run_name "Covid-design" --base_dir "./outputs" --deterministic_seed True --seed "2025" --dtype "fp32" --max_steps "200" --eval_interval "1" --checkpoint_interval "1" --log_interval "1" --iters_to_accumulate "1" --precompute_esm "True" --num_workers "16" --score_input_json_path "$SCORE_INPUT_JSON_PATH" --lr "1e-4" --batchsize "4" --load_params_only "True" --skip_load_optimizer "True" --skip_load_step "True"
