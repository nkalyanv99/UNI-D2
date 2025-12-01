#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
    data.train=text8 \
    data.valid=text8 \
    data.wrap=true \
    data.tokenizer_name_or_path=text8 \
    model=small \
    model.length=256 \
    algo=udlm \
    algo.backbone=dit \
    loader.global_batch_size=512 \
    loader.batch_size=64 \
    loader.eval_global_batch_size=512 \
    loader.eval_batch_size=64 \
    loader.num_workers=16 \
    optim.lr=3e-4 \
    trainer.num_nodes=2 \
    trainer.devices=4 \
    trainer.max_steps=1_000_000 \
    trainer.precision=bf16 \
    trainer.val_check_interval=5_000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    eval.generate_samples=true \
    sampling.num_sample_batches=1 \
    sampling.steps=256 \
    sampling.predictor=ddpm_cache \
    checkpointing.resume_from_ckpt=false \
    wandb.project="final_run" \
    wandb.name="udlm_text8" \
    hydra.run.dir=./outputs/text8/udlm