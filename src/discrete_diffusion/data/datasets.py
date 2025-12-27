"""Specialized dataset builders re-used in the discrete diffusion loader."""

from __future__ import annotations

import json
import os
import shutil
import urllib
import zipfile

import datasets
import fsspec
import numpy as np
import requests
import torch

from .. import utils

LOGGER = utils.get_logger(__name__)

__all__ = [
    "generate_synthetic_dataset",
    "get_lambada_test_dataset",
    "get_text8_dataset",
]


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
  # Special tokens: BOS, EOS, PAD, MASK, UNK (5 total)
  num_special = 5
  num_regular = vocab_size - num_special
  bos_id = num_regular      # BOS token id
  eos_id = num_regular + 1  # EOS token id
  
  dataset = np.zeros((dataset_size, seq_len), dtype=int)
  dataset[:, 0] = bos_id
  dataset[:, -1] = eos_id
  for i in range(dataset_size):
    temp = np.random.randint(num_regular)
    for j in reversed(range(1, seq_len - 1)):
      dataset[i, j] = temp
      if temp != 0:
        temp = temp // 4
      else:
        temp = np.random.randint(num_regular)
  return dataset


def generate_synthetic_dataset(train_dataset_size, validation_dataset_size,
                               seq_len, vocab_size):
  np.random.seed(42)
  train_data = torch.from_numpy(
    _generate_synthetic_data(train_dataset_size, seq_len, vocab_size))
  train_dataset = datasets.Dataset.from_dict({
    'input_ids': train_data,
    'attention_mask': torch.ones_like(train_data),
  })
  train_dataset.set_format(type='torch')

  np.random.seed(41)
  validation_data = torch.from_numpy(
    _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size))
  validation_dataset = datasets.Dataset.from_dict({
    'input_ids': validation_data,
    'attention_mask': torch.ones_like(validation_data),
  })
  validation_dataset.set_format(type='torch')

  return {
    'train': train_dataset,
    'validation': validation_dataset,
  }


def get_lambada_test_dataset():
  url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

  def read_jsonl_to_list(url):
    response = requests.get(url, stream=True)
    data_list = []
    for line in response.iter_lines(decode_unicode=True):
      if line:
        data = json.loads(line)
        data_list.append(data)
    return data_list

  lambada_data = read_jsonl_to_list(url)
  dataset = datasets.Dataset.from_list(lambada_data)
  return dataset


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True,
                      crop_train=False):
  """Adapted from D3PM text datasets."""
  url = 'http://mattmahoney.net/dc/text8.zip'
  cache_dir = f'{cache_dir}/text8' if not crop_train else f'{cache_dir}/text8-crop-train'
  split_names = ['train', 'validation', 'test']
  if not all([
    utils.fsspec_exists(os.path.join(cache_dir, split))
    for split in split_names
  ]):
    raw_cache_dir = os.path.join(cache_dir, 'raw_data')
    if not all([
      utils.fsspec_exists(
        os.path.join(raw_cache_dir, f'text8.{split}.txt'))
      for split in split_names
    ]):
      if not utils.fsspec_exists(os.path.join(raw_cache_dir, 'text8.zip')):
        utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
        LOGGER.info('Downloading text8 from URL %s.', url)
        with (urllib.request.urlopen(url) as in_stream,
              open(os.path.join(raw_cache_dir, 'text8.zip'),
                   'wb') as out_file):
          shutil.copyfileobj(in_stream, out_file)
      with fsspec.open(
        os.path.join(raw_cache_dir, 'text8.zip'),
        'rb') as f:
        rawdata = zipfile.ZipFile(f).read('text8').decode('utf-8')
      splits = {
        'train': rawdata[:90000000],
        'validation': rawdata[90000000: 95000000],
        'test': rawdata[95000000:],
      }
      for split, data in splits.items():
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'w') as f:
          f.write(data)
    else:
      splits = {}
      for split in split_names:
        _path = os.path.join(raw_cache_dir,
                             f'text8.{split}.txt')
        with fsspec.open(_path, 'r') as f:
          splits[split] = f.read()
    def chunks(lst, n):
      for i in range(0, len(lst), n):
        yield lst[i:i + n]
    dataset_dict = {}
    for k, v in splits.items():
      chunk_size = 2 * max_seq_length if (k == 'train' and crop_train) else max_seq_length
      text = list(chunks(v, chunk_size))
      if drop_last and len(text[-1]) < chunk_size:
        text = text[:-1]
      dataset_dict[k] = datasets.Dataset.from_dict({'text': text})
    dataset = datasets.DatasetDict(dataset_dict)
    dataset.save_to_disk(cache_dir)
  else:
    dataset = datasets.load_from_disk(cache_dir)
  return dataset

