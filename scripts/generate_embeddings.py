"""Generate per-token mBERT embeddings for a tokenized text file.

Output layout
-------------
For each sentence (1-based string key matching sentence order in the input
text), an HDF5 dataset of shape (num_layers, num_words, hidden_dim) is
stored, where:
  * num_layers includes the input embeddings (layer 0) and every
    transformer block output (layers 1..L). For mBERT-base, num_layers = 13.
  * num_words is the count of whitespace-separated tokens in the input
    line (after subword aggregation).
  * hidden_dim = 768 for mBERT-base.

This matches the layer indexing convention of Hewitt & Manning (2019):
``model_layer: 0`` selects the input embeddings, ``model_layer: 12`` the
last hidden state.

CLI
---
  python -m scripts.generate_embeddings <text_file> <hdf5_file> [options]

  --model-name MODEL       HuggingFace model id (default: bert-base-multilingual-cased)
  --aggregation MODE       Subword aggregation: 'mean' (default) or 'first'.
                           Hewitt & Manning use 'first'.
  --layers SPEC            Which layers to save: 'all' (default) or a
                           comma-separated list, e.g. '0,7,12'.
  --random-init            Use a randomly initialised model with the same
                           architecture (control baseline).
  --seed N                 Random seed for --random-init weight init.
                           Required for reproducible random baselines.
  --batch-size N           Sentences per forward pass (default: 8).

The output is always 3D (num_layers, num_words, hidden_dim), even when only
one layer is requested. The data loader in data.py auto-detects this and
indexes via the YAML's ``model.model_layer`` field.
"""
from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_MODEL = 'bert-base-multilingual-cased'


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('text_file', help='Whitespace-tokenized text, one sentence per line.')
  p.add_argument('hdf5_file', help='Output HDF5 file.')
  p.add_argument('--model-name', default=DEFAULT_MODEL)
  p.add_argument('--aggregation', choices=['mean', 'first'], default='mean')
  p.add_argument('--layers', default='all',
                 help="'all' or a comma-separated list of layer indices.")
  p.add_argument('--random-init', action='store_true',
                 help='Use a randomly-initialised model (control baseline).')
  p.add_argument('--seed', type=int, default=0,
                 help='Random seed used for --random-init weight '
                      'initialisation. Ignored when not using random init.')
  p.add_argument('--batch-size', type=int, default=8)
  return p.parse_args(argv)


def load_model(model_name: str, random_init: bool, seed: int):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  if random_init:
    # Seed BEFORE constructing the random model so that successive calls
    # with the same --seed produce the exact same baseline.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_config(config)
  else:
    model = AutoModel.from_pretrained(model_name)
  model.eval()
  return tokenizer, model


def select_layer_indices(spec: str, num_layers: int):
  """spec: 'all' or '0,7,12'. Returns sorted unique list of ints."""
  if spec == 'all':
    return list(range(num_layers))
  out = sorted({int(x) for x in spec.split(',') if x.strip() != ''})
  for layer in out:
    if not 0 <= layer < num_layers:
      raise ValueError(f'Layer {layer} out of range [0, {num_layers}).')
  return out


def aggregate_subwords(token_states, word_ids, aggregation: str):
  """Reduce subword vectors to one vector per word.

  Args:
    token_states: numpy array (seq_len, hidden_dim) for one sentence.
    word_ids: list of length seq_len mapping each token to a word id
      (None for special tokens).
    aggregation: 'mean' averages all subword pieces of a word; 'first'
      takes only the first piece (Hewitt & Manning, 2019).

  Returns:
    numpy array (num_words, hidden_dim).
  """
  word_to_pieces: dict[int, list[int]] = {}
  for piece_idx, w_id in enumerate(word_ids):
    if w_id is None:
      continue
    word_to_pieces.setdefault(w_id, []).append(piece_idx)

  ordered_words = sorted(word_to_pieces)
  out = np.empty((len(ordered_words), token_states.shape[-1]),
                 dtype=token_states.dtype)
  for row, w_id in enumerate(ordered_words):
    pieces = word_to_pieces[w_id]
    if aggregation == 'first':
      out[row] = token_states[pieces[0]]
    else:  # mean
      out[row] = token_states[pieces].mean(axis=0)
  return out


def encode_batch(tokenizer, model, sentences, layer_indices, aggregation,
                 device, max_length):
  """Encode a batch of sentences and return a list of (n_layers, n_words, d).

  Truncation is OFF by design: we want alignment between the input words
  and the produced embeddings to be exact. If a sentence exceeds the
  model's max position embeddings, we raise a clear error rather than
  silently dropping tokens. AnCora sentences fit comfortably in mBERT's
  512-token window in practice (max ~120 words ~ ~160 sub-pieces).
  """
  splits = [s.split() for s in sentences]
  encoded = tokenizer(
    splits,
    is_split_into_words=True,
    padding=True,
    truncation=False,
    return_tensors='pt',
  )
  if encoded['input_ids'].shape[1] > max_length:
    raise ValueError(
      f'Batch contains a sentence longer than the model max length '
      f'({encoded["input_ids"].shape[1]} > {max_length}). Reduce '
      f'--batch-size to isolate the offending sentence, then either '
      f'split it manually or skip it.')
  encoded = encoded.to(device)

  with torch.no_grad():
    outputs = model(**encoded, output_hidden_states=True)

  # outputs.hidden_states: tuple of length (num_layers,), each
  # (batch, seq_len, hidden_dim).
  stacked = torch.stack([outputs.hidden_states[l] for l in layer_indices])
  # -> (num_layers_selected, batch, seq_len, hidden_dim)
  stacked = stacked.permute(1, 0, 2, 3).cpu().numpy()
  # -> (batch, num_layers_selected, seq_len, hidden_dim)

  results = []
  for i in range(len(sentences)):
    word_ids = encoded.word_ids(batch_index=i)
    per_layer = []
    for layer in range(stacked.shape[1]):
      per_layer.append(
        aggregate_subwords(stacked[i, layer], word_ids, aggregation))
    per_word = np.stack(per_layer, axis=0)
    results.append(per_word)
  return results


def main(argv=None):
  args = parse_args(argv)

  print(f'Loading model: {args.model_name} '
        f'(random_init={args.random_init}, seed={args.seed})')
  tokenizer, model = load_model(args.model_name, args.random_init, args.seed)
  num_layers = model.config.num_hidden_layers + 1  # +1 for input embeddings
  layer_indices = select_layer_indices(args.layers, num_layers)
  print(f'Will save layers: {layer_indices}')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  print(f'Device: {device}')
  max_length = model.config.max_position_embeddings

  with open(args.text_file, 'r', encoding='utf-8') as f:
    sentences = [line.rstrip('\n') for line in f if line.strip()]
  print(f'{len(sentences)} sentences in {args.text_file}.')

  with h5py.File(args.hdf5_file, 'w') as f_out:
    f_out.attrs['model_name'] = args.model_name
    f_out.attrs['aggregation'] = args.aggregation
    f_out.attrs['layer_indices'] = layer_indices
    f_out.attrs['random_init'] = args.random_init
    f_out.attrs['seed'] = args.seed

    for start in tqdm(range(0, len(sentences), args.batch_size),
                      desc='[encoding]'):
      batch = sentences[start:start + args.batch_size]
      reps = encode_batch(tokenizer, model, batch, layer_indices,
                          args.aggregation, device, max_length)
      for offset, per_word in enumerate(reps):
        sentence_words = batch[offset].split()
        if per_word.shape[1] != len(sentence_words):
          raise RuntimeError(
            f'Sentence {start + offset}: produced {per_word.shape[1]} '
            f'word vectors for an input of {len(sentence_words)} '
            f'whitespace-separated words. Subword aggregation went '
            f'wrong; check the tokenizer.')
        f_out.create_dataset(str(start + offset), data=per_word,
                             compression='gzip', compression_opts=4)
  print(f'Wrote {args.hdf5_file}.')


if __name__ == '__main__':
  main(sys.argv[1:])
