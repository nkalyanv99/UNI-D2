#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
    algo=candi \
    model=small_candi \
    model.length=1024 \
    model.max_tokens=1024 \
    data=openwebtext-split \
    loader.batch_size=32 \
    loader.eval_batch_size=32 \
    loader.num_workers=16 \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    trainer.max_steps=200000 \
    trainer.val_check_interval=5000 \
    trainer.log_every_n_steps=100 \
    optim.lr=3e-4 \
    lr_scheduler=constant_warmup \
    lr_scheduler.num_warmup_steps=2500 \
    training.loss_precision=float32 \
    training.torch_compile=true \
    noise.r_min=0.05 \
    noise.r_max=0.25 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5000 \
    callbacks.sample_saver.enabled=true \
    checkpointing.resume_from_ckpt=false \
    wandb.project="UNI-D2" \
    wandb.name="candi_owt" \
    hydra.run.dir=./outputs/owt/candi
