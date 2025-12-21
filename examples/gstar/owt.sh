#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

# Pretrained MDLM checkpoint to load as frozen backbone
MDLM_CHECKPOINT="${REPO_ROOT}/outputs/owt/mdlm_finetune_len128/checkpoints/6-100000.ckpt"

if [ ! -f "$MDLM_CHECKPOINT" ]; then
    echo "Error: MDLM checkpoint not found at $MDLM_CHECKPOINT"
    exit 1
fi

echo "Training GStar with frozen MDLM from: $MDLM_CHECKPOINT"

python -u -m discrete_diffusion \
    data=openwebtext-split \
    data.wrap=False \
    data.cache_dir=/home/ubuntu/.cache/huggingface/datasets \
    model=small \
    model.length=128 \
    algo=gstar \
    sampling=gstar \
    training.finetune_path="$MDLM_CHECKPOINT" \
    ++training.strict_load=False \
    training.torch_compile=false \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.num_workers=4 \
    trainer.num_nodes=1 \
    trainer.devices=1 \
    trainer.max_steps=50_000 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.monitor=val/f1 \
    callbacks.checkpoint_monitor.mode=max \
    callbacks.checkpoint_monitor.save_top_k=1 \
    callbacks.sample_saver.enabled=false \
    checkpointing.resume_from_ckpt=false \
    wandb.project="gstar" \
    wandb.name="gstar_owt_len128" \
    hydra.run.dir=./outputs/owt/gstar_len128


