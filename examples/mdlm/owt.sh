#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
    data=openwebtext-split \
    model=small \
    algo=mdlm \
    training.torch_compile=true \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=8 \
    trainer.devices=4 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=10_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    callbacks.sample_saver.enabled=true \
    callbacks.sample_saver.every_n_steps=1000 \
    callbacks.sample_saver.num_samples=5 \
    callbacks.sample_saver.save_dir=./outputs/owt/mdlm/samples \
    checkpointing.resume_from_ckpt=false \
    wandb.project="final_run" \
    wandb.name="mdlm_owt" \
    hydra.run.dir=./outputs/owt/mdlm
