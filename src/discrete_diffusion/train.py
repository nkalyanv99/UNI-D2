"""Public training API for discrete diffusion models."""

import os

import hydra
import lightning as L
import omegaconf
import torch
import torch.serialization

from .data import get_dataloaders, get_tokenizer
from . import utils


def train(config):
  """Main training API.
  
  Args:
    config: Hydra DictConfig or config object with training parameters.
    
  Returns:
    None. Model checkpoints are saved according to config.checkpointing.
  """
  # Add safe globals for checkpoint loading compatibility with PyTorch 2.6+
  # This allows loading checkpoints that contain numpy dtypes
  try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])
  except (ImportError, AttributeError):
    pass  # Older PyTorch versions don't need this
  
  # Set matmul precision to 'high' (TF32) to match FlexMDM
  torch.set_float32_matmul_precision("high")
  
  logger = utils.get_logger(__name__)
  logger.info('Starting Training.')
  
  tokenizer = get_tokenizer(config)
  algo_cls = hydra.utils.get_class(config.algo._target_)
  
  # Ensure dataset processing happens on rank 0 first
  fabric = L.Fabric(num_nodes=config.trainer.num_nodes,
                    devices=config.trainer.devices,
                    accelerator='cuda')
  fabric.launch()
  with fabric.rank_zero_first():
    train_ds, valid_ds = get_dataloaders(config, tokenizer)
  fabric.barrier()
  del fabric
  
  # WandB logger
  wandb_logger = L.pytorch.loggers.WandbLogger(
    config=omegaconf.OmegaConf.to_object(config), **config.wandb
  ) if config.get('wandb', None) is not None else None

  # Resume checkpoint path
  ckpt_path = config.checkpointing.resume_ckpt_path if (
    config.checkpointing.resume_from_ckpt and 
    config.checkpointing.resume_ckpt_path is not None and 
    utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
  ) else None

  # Lightning callbacks
  callbacks = [hydra.utils.instantiate(cb) for _, cb in config.callbacks.items()] if 'callbacks' in config else []

  if config.training.finetune_path != '':
    assert utils.fsspec_exists(config.training.finetune_path)
    strict_load = config.training.get("strict_load", True)
    model = algo_cls.load_from_checkpoint(
      config.training.finetune_path, strict=strict_load, tokenizer=tokenizer, config=config)
  else:
    model = algo_cls(config, tokenizer=tokenizer)

  # Torch compile if enabled
  if omegaconf.OmegaConf.select(config, 'training.torch_compile', default=False):
    logger.info('Compiling LightningModule with torch.compile.')
    model = torch.compile(model)

  if config.training.get('fault_tolerant', False):
    os.environ.setdefault('PL_FAULT_TOLERANT_TRAINING', '1')

  trainer = L.Trainer(
    **config.trainer, default_root_dir=os.getcwd(), callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy), logger=wandb_logger)
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)