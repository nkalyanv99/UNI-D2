#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

# Pretrained MDLM checkpoint to load as frozen backbone
MDLM_CHECKPOINT="${REPO_ROOT}/outputs/owt/mdlm_finetune_len${MAX_LENGTH}/dummy_checkpoints/checkpoints/best.ckpt"
TIME_CONDITIONING="${TIME_CONDITIONING:-false}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"

if [ ! -f "$MDLM_CHECKPOINT" ]; then
    echo "Error: MDLM checkpoint not found at $MDLM_CHECKPOINT"
    exit 1
fi

name_suffix=""
if [ "$TIME_CONDITIONING" == "true" ]; then
    name_suffix+="_time_conditioning"
fi
if [ "$FREEZE_BACKBONE" == "false" ]; then
    name_suffix+="_finetune_backbone"
fi

echo "Running in gstar_len${MAX_LENGTH}${name_suffix}"

echo "Training GStar with MDLM from: $MDLM_CHECKPOINT, time conditioning: $TIME_CONDITIONING, freeze backbone: $FREEZE_BACKBONE"

python -u -m discrete_diffusion \
    data=openwebtext-split \
    data.wrap=False \
    data.cache_dir=/home/ubuntu/.cache/huggingface/datasets \
    model=small \
    model.length=${MAX_LENGTH} \
    algo=gstar \
    algo.time_conditioning="$TIME_CONDITIONING" \
    algo.freeze_backbone="$FREEZE_BACKBONE" \
    sampling=gstar \
    ++sampling.sampler.remasker_schedule=plato \
    ++sampling.sampler.t_on=0.55 \
    ++sampling.sampler.t_off=0.05 \ 
    training.finetune_path="$MDLM_CHECKPOINT" \
    ++training.strict_load=False \
    training.torch_compile=false \
    loader.batch_size=64 \
    loader.eval_batch_size=64 \
    loader.num_workers=8 \
    trainer.num_nodes=1 \
    trainer.devices=8 \
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
    wandb.name="gstar_owt_len${MAX_LENGTH}${name_suffix}" \
    hydra.run.dir=./outputs/owt/gstar_len${MAX_LENGTH}${name_suffix}


