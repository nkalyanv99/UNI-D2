#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
    data=openwebtext-split \
    model=small \
    algo=sedd \
    noise=geometric \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.num_workers=16 \
    trainer.num_nodes=8 \
    trainer.devices=4 \
    trainer.val_check_interval=10000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    checkpointing.resume_from_ckpt=false \
    wandb.project="final_run" \
    wandb.name="sedd_owt" \
    hydra.run.dir=./outputs/owt/sedd