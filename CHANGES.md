# Changes Log

## MDLM Checkpoint Integration

### 1. Downloaded MDLM Checkpoint

Downloaded the pre-trained MDLM checkpoint from the official repository:
- Source: [kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt) (also available via Google Drive link in `submodules/mdlm-fork/README.md`)
- Original checkpoint: `mdlm.ckpt`
- Model: MDLM trained on OpenWebText for 1M steps
- Architecture: Small DiT (768 hidden size, 12 blocks, 12 heads)
- Sequence length: 1024 tokens

### 2. Created Conversion Script

Added `checkpoints/mdlm_to_unid2_format.py` to convert MDLM checkpoint format to UNI-D2 repository format.

**Key conversions:**
- Renamed checkpoint keys to match UNI-D2 naming conventions
- Preserved model weights, EMA weights, and optimizer state
- Converted configuration structure to UNI-D2 format
- Stored tokenizer in checkpoint for compatibility

**Usage:**
```bash
python checkpoints/mdlm_to_unid2_format.py
```

**Output:** `checkpoints/mdlm_full.ckpt` (UNI-D2 compatible format)

### 3. Updated Sample Generation Script

Modified `src/discrete_diffusion/evaluations/generate_samples.py`:

**Changes:**
- **EMA weights activation**: Added explicit EMA weights activation for generation
  ```python
  if hasattr(model, '_eval_mode'):
      print("Activating EMA weights...")
      model._eval_mode()
  ```

- **Tokenizer from checkpoint**: Load tokenizer from checkpoint when available for consistency
  ```python
  if 'tokenizer' in ckpt['hyper_parameters']:
      print("Using tokenizer from checkpoint")
      tokenizer = ckpt['hyper_parameters']['tokenizer']
  ```

- **Special tokens visibility**: Changed `skip_special_tokens=False` to make `<|endoftext|>` tokens visible in output (later reverted to `True` based on user preference)

### 4. Sample Generation Results

Generated samples using the converted checkpoint with the following command:
```bash
python src/discrete_diffusion/evaluations/generate_samples.py \
    checkpoint_path=checkpoints/mdlm_full.ckpt \
    num_samples=8 \
    batch_size=8 \
    samples_path=samples_mdlm_ema.pt \
    save_text=true \
    device=cuda \
    num_steps=1024
```

**Example Generated Sample:**

```
Sample 0:
 and point that out, you tell me, it's --

J: No, I'm never right here in the field. Actually, you can't see what I do 
after I play in a game like that. I can't spend a lot of time talking about that.

It's the same, it's all consistent. Sometimes you play the game and the situation 
goes wrong and sometimes it doesn't even involve you and you're doing that and 
trying to get away with it, I'll take advantage and force myself to do things I 
know about myself gamewise when I'm playing on the field and it's all about 
seeing what I have so long and it's perfect for me. It is all about looking into 
yourself. You see that when you don't see anybody else's game or when you are 
playing your craft and the game and you don't see that.
```

**Note on Sample Format:**
The samples start mid-sentence because the MDLM model was trained with `wrap=True` data preprocessing, which concatenates documents and chunks them at fixed intervals. This means training sequences can start at any point in a document (not just at natural boundaries), and the model learned to generate from this distribution. This is expected behavior and matches the training setup described in the MDLM paper.

### Files Modified/Created

**Created:**
- `checkpoints/mdlm_to_unid2_format.py` - Conversion script
- `checkpoints/mdlm_full.ckpt` - Converted checkpoint (not in repo, generated locally)
- `samples_mdlm_ema.pt` - Generated samples tensor
- `samples_mdlm_ema.txt` - Generated samples as text

**Modified:**
- `src/discrete_diffusion/evaluations/generate_samples.py` - Added EMA activation and checkpoint tokenizer loading

