"""Script to generate samples from a trained checkpoint.

This script loads a trained model checkpoint and generates samples using the
configured sampler. The output is saved as a PyTorch tensor (.pt) which can be
used for evaluation (e.g. with generative_ppl.py).

Supports multi-GPU parallel generation by dividing samples across available GPUs.
Also supports CPU-only generation with device=cpu.
"""

import hydra
import torch
import torch.multiprocessing as mp
import tqdm
from pathlib import Path
from omegaconf import OmegaConf

from discrete_diffusion.data import get_tokenizer


def get_num_devices(cfg):
    """Determine the number of devices to use for generation."""
    # Check if devices is explicitly set in CLI config
    if cfg.get("devices", None) is not None:
        return cfg.devices
    
    # Default to 1 device
    return 1


def worker_generate(rank, num_devices, samples_per_device, remainder, cfg, checkpoint_path, 
                    model_config, tokenizer, return_dict, use_cpu=False):
    """Worker function for generating samples on a specific GPU or CPU.
    
    Args:
        rank: Device rank (0 to num_devices-1)
        num_devices: Total number of devices
        samples_per_device: Base number of samples per device
        remainder: Extra samples to distribute among first devices
        cfg: Generation config
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration
        tokenizer: Tokenizer instance
        return_dict: Shared dict for returning results
        use_cpu: Whether to use CPU instead of GPU
    """
    try:
        if use_cpu:
            device = torch.device("cpu")
            print(f"[CPU] Loading model and generating samples...")
        else:
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            # Log which physical GPU we're using
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(rank)
                print(f"[Device {rank}] Using GPU: {gpu_name}")
        
        torch.set_float32_matmul_precision('high')
        torch.set_grad_enabled(False)
        
        # Calculate samples for this worker
        # Distribute remainder among first 'remainder' devices
        num_samples_this_device = samples_per_device + (1 if rank < remainder else 0)
        
        if num_samples_this_device == 0:
            return_dict[rank] = torch.tensor([], dtype=torch.long)
            return
        
        device_label = "CPU" if use_cpu else f"Device {rank}"
        print(f"[{device_label}] Loading model and generating {num_samples_this_device} samples...")
        
        # Load model on this device
        algo_target = model_config.algo._target_
        algo_cls = hydra.utils.get_class(algo_target)
        
        model = algo_cls.load_from_checkpoint(
            checkpoint_path, 
            config=model_config, 
            tokenizer=tokenizer,
            map_location=device,
            strict=False
        )
        
        model.to(device)
        model.eval()
        
        # Activate EMA weights for generation
        if hasattr(model, '_eval_mode'):
            model._eval_mode()
        
        if cfg.torch_compile and not use_cpu:
            model = torch.compile(model)
        
        batch_size = cfg.batch_size
        num_steps = cfg.num_steps
        
        all_samples = []
        
        # Generate samples for this device
        samples_generated = 0
        while samples_generated < num_samples_this_device:
            current_batch_size = min(batch_size, num_samples_this_device - samples_generated)
            
            samples = model.generate_samples(
                num_samples=current_batch_size,
                num_steps=num_steps
            )
            
            all_samples.append(samples.detach().cpu())
            samples_generated += current_batch_size
        
        result = torch.cat(all_samples, dim=0) if all_samples else torch.tensor([], dtype=torch.long)
        print(f"[{device_label}] Generated {len(result)} samples")
        return_dict[rank] = result
        
    except Exception as e:
        device_label = "CPU" if use_cpu else f"Device {rank}"
        print(f"[{device_label}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = None


@hydra.main(config_path="../../../configs", config_name="generate_samples", version_base="1.3")
def main(cfg):
    # Check if we should use CPU
    use_cpu = cfg.get("device", "cuda") == "cpu"
    
    if use_cpu:
        print("Running in CPU mode...")
    elif not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU mode.")
        use_cpu = True
    
    torch.set_float32_matmul_precision('high')

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint to extract config (on CPU first)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config from hyper_parameters
    if 'hyper_parameters' not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'. Cannot load config.")
    
    if 'config' not in ckpt['hyper_parameters']:
         raise ValueError("Checkpoint hyper_parameters does not contain 'config'.")
         
    model_config = ckpt['hyper_parameters']['config']
    # Ensure it's an OmegaConf object
    if not isinstance(model_config, (dict, list, OmegaConf.get_type("DictConfig"), OmegaConf.get_type("ListConfig"))):
         model_config = OmegaConf.create(model_config)
    
    # Override sampling config if provided in CLI arguments
    if cfg.get("sampling", None) is not None:
        OmegaConf.set_struct(model_config.sampling, False)
        model_config.sampling = OmegaConf.merge(model_config.sampling, cfg.sampling)
        OmegaConf.set_struct(model_config.sampling, True)
        print(f"Sampling config overridden: {cfg.sampling.sampler._target_}")
    
    # Get tokenizer - use the one from checkpoint to ensure compatibility
    print("Loading tokenizer...")
    if 'tokenizer' in ckpt['hyper_parameters']:
        print("Using tokenizer from checkpoint")
        tokenizer = ckpt['hyper_parameters']['tokenizer']
    else:
        print("Creating new tokenizer from config")
        tokenizer = get_tokenizer(model_config)
    
    # Determine number of devices
    num_devices = get_num_devices(cfg)
    
    if use_cpu:
        # CPU mode - single device only
        num_devices = 1
    else:
        available_devices = torch.cuda.device_count()
        print(f"Available CUDA devices: {available_devices}")
        
        if num_devices > available_devices:
            print(f"Warning: Requested {num_devices} devices but only {available_devices} available. Using {available_devices}.")
            num_devices = available_devices
        
        # Ensure we have at least 1 device
        num_devices = max(1, num_devices)
    
    num_samples = cfg.num_samples
    
    # Calculate samples per device
    samples_per_device = num_samples // num_devices
    remainder = num_samples % num_devices
    
    print(f"Detected algorithm class: {hydra.utils.get_class(model_config.algo._target_).__name__}")
    
    if use_cpu:
        print(f"Generating {num_samples} samples on CPU")
    else:
        print(f"Generating {num_samples} samples across {num_devices} GPU(s)")
        print(f"  - Base samples per GPU: {samples_per_device}")
        if remainder > 0:
            print(f"  - Extra samples on first {remainder} GPU(s): +1 each")
    
    # Free memory from checkpoint loading
    del ckpt
    
    if num_devices == 1:
        # Single device mode - no multiprocessing needed
        if use_cpu:
            print("Running in single-CPU mode...")
        else:
            print("Running in single-GPU mode...")
        manager = mp.Manager()
        return_dict = manager.dict()
        worker_generate(0, 1, samples_per_device, remainder, cfg, checkpoint_path, 
                       model_config, tokenizer, return_dict, use_cpu=use_cpu)
        all_samples = [return_dict[0]]
    else:
        # Multi-GPU mode using multiprocessing
        print("Spawning worker processes...")
        mp.set_start_method('spawn', force=True)
        
        manager = mp.Manager()
        return_dict = manager.dict()
        
        processes = []
        for rank in range(num_devices):
            p = mp.Process(
                target=worker_generate,
                args=(rank, num_devices, samples_per_device, remainder, cfg, 
                      checkpoint_path, model_config, tokenizer, return_dict, use_cpu)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results in order
        all_samples = []
        for rank in range(num_devices):
            if return_dict.get(rank) is None:
                raise RuntimeError(f"Worker on device {rank} failed to generate samples")
            if len(return_dict[rank]) > 0:
                all_samples.append(return_dict[rank])
    
    # Concatenate all samples
    if all_samples:
        all_samples = torch.cat(all_samples, dim=0)
    else:
        all_samples = torch.tensor([], dtype=torch.long)
    
    print(f"Total samples collected: {len(all_samples)}")
    
    # Verify we have exactly the requested number of samples
    if len(all_samples) != num_samples:
        print(f"Warning: Expected {num_samples} samples but got {len(all_samples)}")
    
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
