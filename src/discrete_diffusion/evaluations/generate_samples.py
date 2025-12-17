"""Script to generate samples from a trained checkpoint.

This script loads a trained model checkpoint and generates samples using the
configured sampler. The output is saved as a PyTorch tensor (.pt) which can be
used for evaluation (e.g. with generative_ppl.py).
"""

import hydra
import torch
import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from discrete_diffusion.data import get_tokenizer

@hydra.main(config_path="../../../configs/eval", config_name="generate_samples", version_base="1.3")
def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config from hyper_parameters
    if 'hyper_parameters' not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'. Cannot load config.")
    
    if 'config' not in ckpt['hyper_parameters']:
         raise ValueError("Checkpoint hyper_parameters does not contain 'config'.")
         
    model_config = ckpt['hyper_parameters']['config']
    # Ensure it's an OmegaConf object
    if not isinstance(model_config, (dict, list, OmegaConf.get_type("DictConfig"), OmegaConf.get_type("ListConfig"))):
         model_config = OmegaConf.create(model_config)
    
    # Get tokenizer - use the one from checkpoint to ensure compatibility
    print("Loading tokenizer...")
    if 'tokenizer' in ckpt['hyper_parameters']:
        print("Using tokenizer from checkpoint")
        tokenizer = ckpt['hyper_parameters']['tokenizer']
    else:
        print("Creating new tokenizer from config")
        tokenizer = get_tokenizer(model_config)
    
    # Identify algorithm class
    algo_target = model_config.algo._target_
    algo_cls = hydra.utils.get_class(algo_target)
    print(f"Detected algorithm class: {algo_cls.__name__}")

    # Load model
    # We need to pass the config and tokenizer as they are init arguments
    print("Loading model...")
    model = algo_cls.load_from_checkpoint(
        checkpoint_path, 
        config=model_config, 
        tokenizer=tokenizer,
        map_location=device
    )
    
    model.to(device)
    model.eval()
    
    # IMPORTANT: Activate EMA weights for generation
    if hasattr(model, '_eval_mode'):
        print("Activating EMA weights...")
        model._eval_mode()
    
    if cfg.torch_compile:
        print("Compiling model...")
        model = torch.compile(model)

    num_samples = cfg.num_samples
    batch_size = cfg.batch_size
    num_steps = cfg.num_steps
    
    print(f"Generating {num_samples} samples (batch_size={batch_size}, steps={num_steps or 'default'})")
    
    all_samples = []
    
    # Progress bar
    with tqdm.tqdm(total=num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        for i in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - i)
            
            # Generate samples
            # We use model.generate_samples which delegates to the configured sampler
            samples = model.generate_samples(
                num_samples=current_batch_size,
                num_steps=num_steps
            )
            
            all_samples.append(samples.detach().cpu())
            pbar.update(current_batch_size)
            
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save samples
    out_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(all_samples, out_path)
    print(f"Saved {len(all_samples)} samples to {out_path}")

    if cfg.get("save_text", False):
        print("Decoding samples to text...")
        texts = tokenizer.batch_decode(all_samples, skip_special_tokens=True)
        text_path = out_path.with_suffix('.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"Sample {i}:\n{text}\n{'-'*80}\n")
        print(f"Saved text samples to {text_path}")

if __name__ == "__main__":
    main()

