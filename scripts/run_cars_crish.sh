#!/usr/bin/env bash
set -eu
python train.py \
    --dataset_name scars --batch_size 128 --grad_from_block 11 --epochs 200 \
    --num_workers 8 --use_ssb_splits --sup_weight 0.35 --weight_decay 5e-5 \
    --transform imagenet --lr 0.1 --eval_funcs v2 --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --memax_weight 1 \
    --crish_mode E3 --crish_warmup 10 --crish_every 1 --crish_evidence 512 \
    --crish_trials 2 --crish_k 5 --crish_boundaries 64 --crish_scoring 256 \
    --crish_probe_known 32 --crish_probe_unknown 64 --crish_cap 8 \
    --crish_confirmations 8 --crish_seconds 300 --run_seed 1 --exp_name cars_crish "$@"
