"""Standalone Generative Perplexity and Diversity evaluation.

Loads generated samples and evaluates NLL/PPL with a chosen eval LM.
Also computes diversity metrics using the diversity library.
Supports multi-GPU parallel evaluation by dividing samples across GPUs.

Supported sample formats:
- .pt: torch.Tensor of token ids (shape [N, T] or [N, 1, T])
- .npz: numpy array under key 'samples' (shape [N, T])
- .json: contains base64-encoded numpy array under key 'np_tokens_b64'

This script decodes using `model_tokenizer` and (optionally) retokenizes with
the eval model's tokenizer before computing loss.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer


from typing import List
import nltk

def ngram_diversity_score(
        data: List[str],
        num_n: int = 4, 
) -> float:
    """ Calculates corpus-level ngram diversity based on unique ngrams 
       (e.g., https://arxiv.org/pdf/2202.00666.pdf).

    Args:
        data (List[str]): List of documents. 
        num_n (int): Max ngrams to test up to. Defaults to 5. 

    Returns:
        float: ngram diveristy score.
    """
    score = 1.
    data = ' '.join(data).split(' ') # format to list of words

    for i in range(2, num_n + 1): 
        ngrams = list(nltk.ngrams(data, i))
        # num unique ngrams / all ngrams for each size n 
        score *= len(set(ngrams)) / len(ngrams) 

    return round(score, 3)


def _load_samples(samples_path: str) -> np.ndarray:
  path = Path(hydra.utils.to_absolute_path(samples_path))
  if not path.exists():
    raise FileNotFoundError(f"Samples not found at {path}")

  if path.suffix == ".pt":
    z_ts = torch.load(path, weights_only=True)
    if isinstance(z_ts, torch.Tensor):
      arr = z_ts.detach().cpu()
    else:
      # if saved as dict or list, try common keys/shapes
      raise ValueError("Unsupported .pt structure; expected a Tensor.")
    if arr.ndim == 3 and arr.shape[1] == 1:
      arr = arr.squeeze(1)
    return arr.numpy()

  if path.suffix == ".npz":
    content = np.load(path)
    if 'samples' not in content:
      raise KeyError(".npz must contain 'samples' key")
    return content['samples']

  if path.suffix == ".json":
    from ..utils import utils as _utils
    with open(path, 'r') as f:
      payload = json.load(f)
    if 'np_tokens_b64' not in payload:
      raise KeyError(".json must contain 'np_tokens_b64' key")
    arr = _utils.base64_to_np(payload['np_tokens_b64'])
    return arr

  raise ValueError(f"Unsupported samples format: {path.suffix}")


def _retokenize(
  texts: List[str],
  tokenizer: AutoTokenizer,
  max_length: int,
  device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
  # Default context windows for common models; conservative fallback
  eval_context_size = 4096 if 'llama' in tokenizer.name_or_path.lower() else 1024
  batch = tokenizer(
    texts,
    return_tensors="pt",
    return_token_type_ids=False,
    return_attention_mask=True,
    truncation=True,
    padding=True,
    max_length=max_length,
  )
  attn_mask = batch['attention_mask'].to(device)
  input_ids = batch['input_ids'].to(device)
  return input_ids, attn_mask, eval_context_size


def compute_diversity_metrics(texts: List[str], n: int = 4) -> dict:
    """Compute diversity metrics for a list of texts.
    
    Args:
        texts: List of text samples
        n: N-gram size for diversity metrics (default: 4)
    
    Returns:
        Dictionary containing diversity metrics
    """
    print(f"Computing distinct-{n} diversity metric...")
    
    # N-gram diversity: ratio of unique n-grams to total n-grams (higher = more diverse)
    score = float(ngram_diversity_score(texts, num_n=n))
    
    return {"distinct_n": score}


def get_num_devices(cfg):
    """Determine the number of devices to use for evaluation."""
    if cfg.get("devices", None) is not None:
        return cfg.devices
    # Default to 1 device
    return 1


def worker_evaluate(rank, num_devices, texts_chunk, cfg, return_dict, use_cpu=False):
    """Worker function for evaluating samples on a specific GPU or CPU.
    
    Args:
        rank: Device rank (0 to num_devices-1)
        num_devices: Total number of devices
        texts_chunk: List of text samples to evaluate on this device
        cfg: Evaluation config
        return_dict: Shared dict for returning results
        use_cpu: Whether to use CPU instead of GPU
    """
    try:
        if use_cpu:
            device = torch.device("cpu")
            device_label = "CPU"
        else:
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(rank)
                print(f"[Device {rank}] Using GPU: {gpu_name}")
            device_label = f"Device {rank}"
        
        torch.set_float32_matmul_precision('high')
        torch.set_grad_enabled(False)
        
        if len(texts_chunk) == 0:
            return_dict[rank] = {
                'total_nll': 0.0,
                'total_acc': 0.0,
                'total_tokens': 0.0,
                'all_nlls': []
            }
            return
        
        print(f"[{device_label}] Loading eval model and evaluating {len(texts_chunk)} samples...")
        
        # Load eval model on this device
        if use_cpu:
            eval_model = AutoModelForCausalLM.from_pretrained(cfg.pretrained_model)
            eval_model.to(device)
        else:
            eval_model = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_model, 
                device_map={"": device}
            )
        
        eval_tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
        if eval_tokenizer.pad_token_id is None:
            eval_tokenizer.pad_token = eval_tokenizer.eos_token
        
        if cfg.torch_compile and not use_cpu:
            eval_model = torch.compile(eval_model)
        
        total_acc = 0.0
        total_nll = 0.0
        total_tokens = 0.0
        all_nlls: List[float] = []
        
        with torch.no_grad():
            for i in range(0, len(texts_chunk), cfg.batch_size):
                xs = texts_chunk[i:i + cfg.batch_size]
                
                input_ids, attn_mask, context_size = _retokenize(
                    xs, eval_tokenizer, cfg.max_length, device)
                
                # Evaluate possibly only the first chunk up to EOS
                logits = eval_model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits[:, :-1]
                labels = input_ids[:, 1:]
                loss_mask = attn_mask[:, :-1]
                
                nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
                
                if cfg.first_chunk_only:
                    eos_id = eval_tokenizer.eos_token_id
                    eos_mask = (labels == eos_id).cumsum(-1) == 0  # valid until first EOS (exclusive)
                    valid = loss_mask.bool() & eos_mask
                else:
                    valid = loss_mask.bool()
                
                valid = valid.to(nll.dtype)
                all_nlls.extend(nll[valid == 1].detach().cpu().numpy().tolist())
                total_nll += float((nll * valid).sum().item())
                
                acc = (logits.argmax(-1) == labels).to(nll.dtype)
                total_acc += float((acc * valid).sum().item())
                total_tokens += float(valid.sum().item())
        
        print(f"[{device_label}] Evaluated {len(texts_chunk)} samples, {int(total_tokens)} tokens")
        
        return_dict[rank] = {
            'total_nll': total_nll,
            'total_acc': total_acc,
            'total_tokens': total_tokens,
            'all_nlls': all_nlls
        }
        
    except Exception as e:
        device_label = "CPU" if use_cpu else f"Device {rank}"
        print(f"[{device_label}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = None


@hydra.main(config_path='../../../configs', config_name='gen_ppl', version_base='1.3')
def main(cfg):
    # Check if we should use CPU
    use_cpu = cfg.get("device", "cuda") == "cpu"
    
    if use_cpu:
        print("Running in CPU mode...")
    elif not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU mode.")
        use_cpu = True
    
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    # Decode tokens (from diffusion model) to text using its tokenizer
    print(f"Loading model tokenizer: {cfg.model_tokenizer}")
    model_tokenizer = AutoTokenizer.from_pretrained(cfg.model_tokenizer)

    # Load samples and make text
    print(f"Loading samples from: {cfg.samples_path}")
    z_ts = _load_samples(cfg.samples_path)
    if z_ts.ndim != 2:
        raise ValueError(f"Expected 2D [N, T] tokens array, got {z_ts.shape}")
    texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)
    
    print(f"Loaded {len(texts)} samples")
    
    # Compute diversity metrics with n=4
    diversity_n = cfg.get("diversity_n", 4)
    diversity_metrics = compute_diversity_metrics(texts, n=diversity_n)
    
    # Determine number of devices
    num_devices = get_num_devices(cfg)
    
    if use_cpu:
        num_devices = 1
    else:
        available_devices = torch.cuda.device_count()
        print(f"Available CUDA devices: {available_devices}")
        
        if num_devices > available_devices:
            print(f"Warning: Requested {num_devices} devices but only {available_devices} available. Using {available_devices}.")
            num_devices = available_devices
        
        num_devices = max(1, num_devices)
    
    # Split texts across devices
    samples_per_device = len(texts) // num_devices
    remainder = len(texts) % num_devices
    
    text_chunks = []
    start_idx = 0
    for i in range(num_devices):
        chunk_size = samples_per_device + (1 if i < remainder else 0)
        text_chunks.append(texts[start_idx:start_idx + chunk_size])
        start_idx += chunk_size
    
    if use_cpu:
        print(f"Evaluating {len(texts)} samples on CPU")
    else:
        print(f"Evaluating {len(texts)} samples across {num_devices} GPU(s)")
        for i, chunk in enumerate(text_chunks):
            print(f"  - Device {i}: {len(chunk)} samples")
    
    if num_devices == 1:
        # Single device mode - no multiprocessing needed
        if use_cpu:
            print("Running in single-CPU mode...")
        else:
            print("Running in single-GPU mode...")
        manager = mp.Manager()
        return_dict = manager.dict()
        worker_evaluate(0, 1, text_chunks[0], cfg, return_dict, use_cpu=use_cpu)
        results = [return_dict[0]]
    else:
        # Multi-GPU mode using multiprocessing
        print("Spawning worker processes...")
        mp.set_start_method('spawn', force=True)
        
        manager = mp.Manager()
        return_dict = manager.dict()
        
        processes = []
        for rank in range(num_devices):
            p = mp.Process(
                target=worker_evaluate,
                args=(rank, num_devices, text_chunks[rank], cfg, return_dict, use_cpu)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results
        results = []
        for rank in range(num_devices):
            if return_dict.get(rank) is None:
                raise RuntimeError(f"Worker on device {rank} failed to evaluate samples")
            results.append(return_dict[rank])
    
    # Aggregate results from all workers
    total_nll = sum(r['total_nll'] for r in results)
    total_acc = sum(r['total_acc'] for r in results)
    total_tokens = sum(r['total_tokens'] for r in results)
    all_nlls = []
    for r in results:
        all_nlls.extend(r['all_nlls'])

    if total_tokens == 0:
        raise RuntimeError("No valid tokens for evaluation (check inputs/EOS handling)")

    avg_nll = total_nll / total_tokens
    ppl = float(np.exp(avg_nll))
    acc = total_acc / total_tokens

    metrics = {
        "file": Path(cfg.samples_path).stem,
        "pretrained_model": cfg.pretrained_model,
        "median_nll": float(np.median(all_nlls)) if all_nlls else float('nan'),
        "avg_nll": float(avg_nll),
        "ppl": float(ppl),
        "acc": float(acc),
        "tokens": int(total_tokens),
        "retokenize": bool(cfg.retokenize),
        "first_chunk_only": bool(cfg.first_chunk_only),
        "num_samples": len(texts),
        "num_devices": num_devices,
        # Diversity metrics (n=4 by default)
        "diversity_n": diversity_n,
        **diversity_metrics,
    }

    print(json.dumps(metrics, indent=2))
    if cfg.get("metrics_path"):
        metrics_path = cfg.metrics_path
    else:
        # Replace samples/ with metrics/ and extension with .json
        metrics_path = Path(cfg.samples_path.replace('samples/', 'metrics/', 1)).with_suffix(".json")
    out_path = Path(hydra.utils.to_absolute_path(metrics_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(metrics, f)
    print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
  main()
