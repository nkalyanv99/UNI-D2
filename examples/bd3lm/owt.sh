#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

BLOCK_SIZE=${BLOCK_SIZE:-16}

export BLOCK_SIZE
export PYTHONPATH="src:${PYTHONPATH:-}"
export DEBUG_TRACE=0

python -u -m discrete_diffusion \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.global_batch_size=512 \
    loader.num_workers=2 \
    trainer.num_nodes=8 \
    trainer.devices=4 \
    trainer.accumulate_grad_batches=1 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=100 \
    trainer.max_steps=1_000_000 \
    model=block_dit \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=openwebtext-split \
    data.insert_train_eos=True \
    data.insert_valid_eos=True \
    data.insert_train_special=True \
    data.insert_valid_special=True \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    wandb.project="final_run" \
    wandb.name=bd3lm_owt \
    model.attn_backend=flex \
    training.resample=True \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    checkpointing.resume_from_ckpt=false \
    hydra.run.dir=./outputs/owt/bd3lm_block${BLOCK_SIZE}_debug \
    algo.ignore_bos=True \
    model.adaln=False \
    model.tie_word_embeddings=True