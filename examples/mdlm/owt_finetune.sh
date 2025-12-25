#!/usr/bin/env bash
set -euo pipefail

# Fine-tune MDLM checkpoint with padding-based training
# Usage: MAX_LENGTH=128 bash examples/mdlm/owt_finetune.sh
# Default MAX_LENGTH: 1024

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

# Path to converted checkpoint (use absolute path)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/mdlm_full.ckpt}"

# Max sequence length (default 1024)
MAX_LENGTH="${MAX_LENGTH:-1024}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo "Please run: python checkpoints/mdlm_to_unid2_format.py"
    exit 1
fi

echo "Fine-tuning from checkpoint: $CHECKPOINT_PATH"
echo "Max sequence length: $MAX_LENGTH"
echo "Time conditioning: $TIME_CONDITIONING"

python -u -m discrete_diffusion \
    data=openwebtext-split \
    data.wrap=False \
    data.cache_dir=/home/ubuntu/.cache/huggingface/datasets \
    model=small \
    model.length=$MAX_LENGTH \
    algo=mdlm \
    algo.time_conditioning=$TIME_CONDITIONING \
    training.finetune_path="$CHECKPOINT_PATH" \
    training.torch_compile=false \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    loader.global_batch_size=512 \
    loader.num_workers=32 \
    trainer.num_nodes=1 \
    trainer.devices=8 \
    trainer.max_steps=100_000 \
    trainer.val_check_interval=1000 \
    trainer.log_every_n_steps=100 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=5_000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    callbacks.checkpoint_every_n_steps.save_last=true \
    callbacks.checkpoint_monitor.save_top_k=1 \
    callbacks.sample_saver.enabled=true \
    callbacks.sample_saver.every_n_steps=1000 \
    callbacks.sample_saver.num_samples=5 \
    callbacks.sample_saver.save_dir=./outputs/owt/mdlm_finetune_len${MAX_LENGTH}/samples \
    checkpointing.resume_from_ckpt=false \
    wandb.project="gstar" \
    wandb.name="mdlm_owt_finetune_len${MAX_LENGTH}_time_conditioning_${TIME_CONDITIONING}" \
    hydra.run.dir=./outputs/owt/mdlm_finetune_len${MAX_LENGTH}_time_conditioning_${TIME_CONDITIONING}

