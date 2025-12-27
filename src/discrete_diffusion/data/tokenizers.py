"""Custom tokenizer implementations used by discrete diffusion datasets."""

from __future__ import annotations

from typing import Dict, List
import transformers

__all__ = [
    "SyntheticTokenizer",
    "Text8Tokenizer",
]


class SyntheticTokenizer(transformers.PreTrainedTokenizer):
  """Simple synthetic tokenizer for deterministic experiments."""

  # Number of special tokens: BOS, EOS, PAD, MASK, UNK
  NUM_SPECIAL_TOKENS = 5

  def __init__(
    self,
    vocab_size,
    bos_token="[BOS]",
    eos_token="[EOS]",
    sep_token=None,
    cls_token=None,
    pad_token="[PAD]",
    mask_token="[MASK]",
    unk_token="[UNK]",
    **kwargs):
    
    # Regular tokens: 0 to vocab_size - NUM_SPECIAL_TOKENS - 1
    # Special tokens at the end
    num_regular = vocab_size - self.NUM_SPECIAL_TOKENS
    self.tokens = []
    for i in range(num_regular):
      self.tokens.append(str(i) + " ")
    self._vocab_str_to_int = {
      '[BOS]': num_regular,
      '[EOS]': num_regular + 1,
      '[PAD]': num_regular + 2,
      '[MASK]': num_regular + 3,
      '[UNK]': num_regular + 4,
      **{ch: i for i, ch in enumerate(self.tokens)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> Dict[str, int]:
    return self._vocab_str_to_int



class Text8Tokenizer(transformers.PreTrainedTokenizer):

  def __init__(
    self,
    bos_token='[BOS]',
    eos_token='[EOS]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    pad_token='[PAD]',
    mask_token='[MASK]',
    unk_token='[UNK]',
    **kwargs):
    self.characters = list('abcdefghijklmnopqrstuvwxyz ')
    self._vocab_str_to_int = {
      '[CLS]': 0,
      '[SEP]': 1,
      '[BOS]': 2,
      '[EOS]': 3,
      '[MASK]': 4,
      '[PAD]': 5,
      '[RESERVED]': 6,
      '[UNK]': 7,
      **{ch: i + 8 for i, ch in enumerate(self.characters)}}
    self._vocab_int_to_str = {
      v: k for k, v in self._vocab_str_to_int.items()}
    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      sep_token=sep_token,
      cls_token=cls_token,
      pad_token=pad_token,
      mask_token=mask_token,
      unk_token=unk_token,
      **kwargs)

  @property
  def vocab_size(self) -> int:
    return len(self._vocab_str_to_int)

  def _tokenize(self, text: str, **kwargs) -> List[str]:
    return list(text.lower())

  def _convert_token_to_id(self, token: str) -> int:
    return self._vocab_str_to_int.get(
      token, self._vocab_str_to_int['[UNK]'])

  def _convert_id_to_token(self, index: int) -> str:
    return self._vocab_int_to_str[index]

  def convert_tokens_to_string(self, tokens):
    return ''.join(tokens)

  def get_vocab(self) -> Dict[str, int]:
    return self._vocab_str_to_int
