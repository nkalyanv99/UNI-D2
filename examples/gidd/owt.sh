#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

export PYTHONPATH="src:${PYTHONPATH:-}"

python -m src.discrete_diffusion \
  data=openwebtext \
  model=small \
  algo=gidd \
  lr_scheduler=cosine_decay_warmup \
  noise=geometric \
  trainer.num_nodes=1 trainer.devices=8 \
  trainer.max_steps=500000 \
  trainer.accumulate_grad_batches=2 \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=2000 \
  trainer.limit_val_batches=0 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=2000 \
  trainer.precision=bf16 \
  training.torch_compile=false \
  model.length=512 \
  model.dropout=0.0 \
  optim.lr=5e-4 \
  optim.weight_decay=0.02 \
  optim.beta1=0.9 \
  optim.beta2=0.99 \
  optim.eps=1e-9 \
  lr_scheduler.warmup_t=10000 \
  wandb.name='small-gidd-owt-512' \
  wandb.project=UNI-D2