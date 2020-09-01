#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

# optionally
export ENRO_DIR=/Users/Mehrad/Documents/GitHub/genienlp/tests/en-fa-test/ # Download instructions above
# export WANDB_PROJECT="MT" # optional
export MAX_LEN=200
export BS=4
export GAS=2 # gradient accumulation steps

set -e
set -x

python3 finetune.py \
    --learning_rate=3e-5 \
    --do_train \
    --val_check_interval=0.25 \
    --adam_eps 1e-06 \
    --num_train_epochs 20 --src_lang en_XX --tgt_lang ar_AR \
    --data_dir $ENRO_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size=$BS --eval_batch_size=$BS \
    --task translation \
    --warmup_steps 500 \
    --freeze_embeds \
    --model_name_or_path=facebook/mbart-large-cc25 \
    --cache_dir /Users/Mehrad/Documents/GitHub/genienlp/.embeddings \
    "$@"
