#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

python -u -m discrete_diffusion \
  scratch_dir=/fast/project/HFMI_SynergyUnit/kalyan.nadimpalli/.cache/discrete_diffusion \
  algo=flexmdm-anyorder \
  algo.only_embed_insert=false \
  data=wikitext2 \
  data.wrap=false \
  data.chunking=double_newline \
  data.insert_train_eos=false data.insert_valid_eos=false \
  model=flexmdm_anyorder \
  model.length=512 \
  model.hidden_size=256 \
  model.n_heads=4 \
  model.n_blocks=4 \
  model.cond_dim=64 \
  model.dropout=0.1 \
  lr_scheduler=cosine_decay_warmup \
  noise=linear \
  trainer.num_nodes=1 trainer.devices=8 \
  trainer.max_steps=100000 \
  trainer.accumulate_grad_batches=1 \
  trainer.gradient_clip_val=1.0 \
  loader.global_batch_size=512 \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=null \
  trainer.check_val_every_n_epoch=10 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  callbacks.checkpoint_monitor.monitor=val_loss \
  trainer.precision=32 \
  training.torch_compile=true \
  optim.lr=3e-4 \
  optim.weight_decay=0.03 \
  optim.beta1=0.9 \
  optim.beta2=0.999 \
  optim.eps=1e-8 \
  lr_scheduler.warmup_t=2000 \
  lr_scheduler.warmup_lr_init=3e-10 \
  lr_scheduler.lr_min=0.0 \
  seed=42 \
  training.ema=0.9999 \
  wandb.project=UNI-D2 \
  wandb.name='flexmdm-anyorder-wikitext2-ours'
