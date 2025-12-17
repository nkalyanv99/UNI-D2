#!/usr/bin/env python3
"""Convert downloaded MDLM checkpoint to UNI-D2 format.

This script converts a checkpoint from the original MDLM format to the format
expected by the UNI-D2 codebase, specifically for use with generate_samples.py.
"""

import torch
from omegaconf import OmegaConf
from pathlib import Path
from hydra import compose, initialize_config_dir
import os


def convert_checkpoint(input_path: str, output_path: str):
    """Convert checkpoint from old format to new UNI-D2 format.
    
    Args:
        input_path: Path to the downloaded checkpoint (mdlm.ckpt)
        output_path: Path to save converted checkpoint (mdlm_full.ckpt)
    """
    print(f"Loading checkpoint from {input_path}")
    ckpt = torch.load(input_path, map_location='cpu', weights_only=False)
    
    old_config = ckpt['hyper_parameters']['config']
    tokenizer = ckpt['hyper_parameters']['tokenizer']
    
    print(f"Old config structure detected")
    print(f"  - backbone: {old_config.backbone}")
    print(f"  - diffusion: {old_config.diffusion}")
    print(f"  - model type: {old_config.model.type}")
    
    # Get vocab size from tokenizer
    vocab_size = len(tokenizer)
    print(f"  - vocab_size: {vocab_size}")
    
    # Use Hydra to compose the base configuration
    # This ensures we have all the required fields with proper structure
    config_dir = Path(__file__).parent.parent / "configs"
    config_dir = str(config_dir.absolute())
    
    print(f"\nComposing new config from Hydra configs at: {config_dir}")
    
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        # Compose config with the same settings as the training script
        # Based on examples/mdlm/owt.sh
        new_config = compose(
            config_name="config",
            overrides=[
                "data=openwebtext-split",
                "model=small", 
                "algo=mdlm",
                "noise=log-linear",
                "sampling=default",
                "lr_scheduler=constant_warmup",
                "prior=none",
                "strategy=ddp",
            ]
        )
    
    print(f"Base config composed successfully")
    print(f"  - algo._target_: {new_config.algo._target_}")
    print(f"  - model._target_: {new_config.model._target_}")
    print(f"  - noise._target_: {new_config.noise._target_}")
    
    # Update config with values from the old checkpoint where appropriate
    # Preserve critical settings from the trained checkpoint
    # Convert to regular mode to allow adding/updating fields
    OmegaConf.set_struct(new_config, False)
    
    new_config.data = old_config.data
    # Merge model configs carefully
    for key, value in old_config.model.items():
        if key in new_config.model or key in ['name', 'type', 'hidden_size', 'cond_dim', 'length', 
                                                'n_blocks', 'n_heads', 'scale_by_sigma', 'dropout', 
                                                'tie_word_embeddings']:
            new_config.model[key] = value
    
    new_config.loader = old_config.loader
    new_config.training = old_config.training
    new_config.optim = old_config.optim
    new_config.trainer = old_config.trainer
    new_config.eval = old_config.eval
    new_config.wandb = old_config.get('wandb', new_config.wandb)
    new_config.checkpointing = old_config.get('checkpointing', new_config.checkpointing)
    new_config.callbacks = old_config.get('callbacks', new_config.callbacks)
    new_config.strategy = old_config.get('strategy', new_config.strategy)
    
    # For sampling, keep the new config but update specific fields from old config if they exist
    old_sampling_dict = OmegaConf.to_container(old_config.sampling, resolve=True)
    for key in ['steps']:
        if key in old_sampling_dict:
            new_config.sampling[key] = old_sampling_dict[key]
    # Note: old predictor "subs" is not supported, keep default "ddpm_cache"
    print(f"  - Using predictor: {new_config.sampling.predictor} (old was: {old_sampling_dict.get('predictor', 'N/A')})")
    
    # Update algo config with old parameterization settings if present
    if 'backbone' in old_config:
        new_config.algo.backbone = old_config.backbone
    if 'parameterization' in old_config:
        new_config.algo.parameterization = old_config.parameterization
    if 'time_conditioning' in old_config:
        new_config.algo.time_conditioning = old_config.time_conditioning
    
    print(f"\nMerged old checkpoint settings with base config")
    print(f"  - Final config has 'algo': {'algo' in new_config}")
    print(f"  - Final config has 'model._target_': {'_target_' in new_config.model}")
    print(f"  - Final config has 'noise._target_': {'_target_' in new_config.noise}")
    
    # Get actual vocab size from state_dict to ensure it matches
    actual_vocab_size = ckpt['state_dict']['backbone.vocab_embed.embedding'].shape[0]
    print(f"  - Vocab size from tokenizer: {vocab_size}")
    print(f"  - Vocab size from state_dict: {actual_vocab_size}")
    print(f"  - Using vocab_size: {actual_vocab_size}")
    
    # IMPORTANT: Set mask_token to PAD token position
    # The checkpoint was trained with:
    #   - tokenizer.vocab_size = 50257 (base GPT2, ignores added PAD)
    #   - mask_index = tokenizer.vocab_size = 50257 (the PAD token position!)
    #   - vocab_size = 50258 for model (after +=1)
    #   - Embeddings [50258, 768] where index 50257 serves as BOTH PAD and MASK
    print(f"  - Setting mask_token to PAD token (index {tokenizer.pad_token_id})")
    tokenizer.mask_token = tokenizer.pad_token  # Use PAD as mask
    tokenizer.mask_token_id = tokenizer.pad_token_id  # 50257
    print(f"    len(tokenizer): {len(tokenizer)}")
    print(f"    mask_token: {tokenizer.mask_token} (id={tokenizer.mask_token_id})")
    print(f"    PAD and MASK share the same embedding at index {tokenizer.pad_token_id}")
    
    # Update hyper_parameters
    ckpt['hyper_parameters']['config'] = new_config
    ckpt['hyper_parameters']['tokenizer'] = tokenizer
    ckpt['hyper_parameters']['vocab_size'] = actual_vocab_size
    
    # Fix state_dict keys: the new codebase uses "model." prefix instead of "backbone."
    # DO NOT pad embeddings - keep original size to match tokenizer
    print("  - Duplicating state_dict keys: backbone.* -> model.*")
    
    new_state_dict = {}
    for key, value in ckpt['state_dict'].items():
        new_state_dict[key] = value
        # Add duplicate with "model." prefix if key starts with "backbone."
        if key.startswith('backbone.'):
            new_key = key.replace('backbone.', 'model.', 1)
            new_state_dict[new_key] = value
    ckpt['state_dict'] = new_state_dict
    
    # Remove sampler if present (will be recreated from config)
    if 'sampler' in ckpt:
        print("  - Removing 'sampler' key from checkpoint")
        del ckpt['sampler']
    
    # Save converted checkpoint
    print(f"\nSaving converted checkpoint to {output_path}")
    torch.save(ckpt, output_path)
    print("Conversion complete!")
    
    # Verify the conversion
    print("\nVerifying converted checkpoint...")
    verify_ckpt = torch.load(output_path, map_location='cpu', weights_only=False)
    verify_config = verify_ckpt['hyper_parameters']['config']
    print(f"  ✓ Has 'algo': {'algo' in verify_config}")
    print(f"  ✓ Has 'vocab_size': {'vocab_size' in verify_ckpt['hyper_parameters']}")
    print(f"  ✓ algo._target_: {verify_config.algo._target_}")
    print(f"  ✓ model._target_: {verify_config.model._target_}")
    print(f"  ✓ noise._target_: {verify_config.noise._target_}")
    print(f"  ✓ Has 'p_nucleus' in sampling: {'p_nucleus' in verify_config.sampling}")


if __name__ == "__main__":
    input_path = "checkpoints/mdlm.ckpt"
    output_path = "checkpoints/mdlm_full.ckpt"
    
    convert_checkpoint(input_path, output_path)
