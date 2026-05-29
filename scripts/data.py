"""
This module handles the reading of conllx files and hdf5 embeddings.

Specifies Dataset classes, which offer PyTorch Dataloaders for the
train/dev/test splits.
"""
import os
from collections import namedtuple, defaultdict

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import h5py


class SimpleDataset:
  """Reads conllx files to provide PyTorch Dataloaders.

  Reads the data from conllx files into namedtuple form to keep annotation
  information, and provides PyTorch dataloaders and padding/batch collation
  to provide access to train, dev, and test splits.
  """
  def __init__(self, args, task, vocab={}):
    self.args = args
    self.batch_size = args['dataset']['batch_size']
    self.use_disk_embeddings = args['model']['use_disk']
    self.vocab = vocab
    self.observation_class = self.get_observation_class(self.args['dataset']['observation_fieldnames'])
    self.train_obs, self.dev_obs, self.test_obs = self.read_from_disk()
    self.train_dataset = ObservationIterator(self.train_obs, task)
    self.dev_dataset = ObservationIterator(self.dev_obs, task)
    self.test_dataset = ObservationIterator(self.test_obs, task)

  def read_from_disk(self):
    train_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['train_path'])
    dev_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['dev_path'])
    test_corpus_path = os.path.join(self.args['dataset']['corpus']['root'],
        self.args['dataset']['corpus']['test_path'])
    train_observations = self.load_conll_dataset(train_corpus_path)
    dev_observations = self.load_conll_dataset(dev_corpus_path)
    test_observations = self.load_conll_dataset(test_corpus_path)

    train_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['train_path'])
    dev_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['dev_path'])
    test_embeddings_path = os.path.join(self.args['dataset']['embeddings']['root'],
        self.args['dataset']['embeddings']['test_path'])
    train_observations = self.optionally_add_embeddings(train_observations, train_embeddings_path)
    dev_observations = self.optionally_add_embeddings(dev_observations, dev_embeddings_path)
    test_observations = self.optionally_add_embeddings(test_observations, test_embeddings_path)
    return train_observations, dev_observations, test_observations

  def get_observation_class(self, fieldnames):
    return namedtuple('Observation', fieldnames)

  def generate_lines_for_sent(self, lines):
    buf = []
    for line in lines:
      if line.startswith('#'):
        continue
      if not line.strip():
        if buf:
          yield buf
          buf = []
        else:
          continue
      else:
        buf.append(line.strip())
    if buf:
      yield buf

  def load_conll_dataset(self, filepath):
    observations = []
    lines = (x for x in open(filepath, encoding='utf-8'))
    for buf in self.generate_lines_for_sent(lines):
      conllx_lines = []
      for line in buf:
        parts = line.strip().split('\t')
        # Skip CoNLL-U special rows: contraction ranges (e.g. "13-14")
        # and enhanced empty nodes (e.g. "1.1"). Only proper word rows
        # have integer IDs, which is what mBERT sees and what our gold
        # labels reference.
        if not parts[0].isdigit():
          continue
        conllx_lines.append(parts)

      if not conllx_lines:
        continue
      embeddings = [None for x in range(len(conllx_lines))]
      observation = self.observation_class(*zip(*conllx_lines), embeddings)
      observations.append(observation)
    return observations

  def add_embeddings_to_observations(self, observations, embeddings):
    embedded_observations = []
    for observation, embedding in zip(observations, embeddings):
      embedded_observation = self.observation_class(*(observation[:-1]), embedding)
      embedded_observations.append(embedded_observation)
    return embedded_observations

  def generate_token_embeddings_from_hdf5(self, args, observations, filepath, layer_index):
    hf = h5py.File(filepath, 'r')
    indices = filter(lambda x: x != 'sentence_to_index', list(hf.keys()))
    single_layer_features_list = []
    for index in sorted([int(x) for x in indices]):
      observation = observations[index]
      feature_stack = hf[str(index)]
      single_layer_features = feature_stack[layer_index]
      assert single_layer_features.shape[0] == len(observation.sentence)
      single_layer_features_list.append(single_layer_features)
    return single_layer_features_list

  def integerize_observations(self, observations):
    new_observations = []
    if self.vocab == {}:
      raise ValueError("Cannot replace words with integer ids with an empty vocabulary "
          "(and the vocabulary is in fact empty")
    for observation in observations:
      sentence = tuple([vocab[sym] for sym in observation.sentence])
      new_observations.append(self.observation_class(sentence, *observation[1:]))
    return new_observations

  def get_train_dataloader(self, shuffle=True, use_embeddings=True):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=shuffle)

  def get_dev_dataloader(self, use_embeddings=True):
    return DataLoader(self.dev_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def get_test_dataloader(self, use_embeddings=True):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.custom_pad, shuffle=False)

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    return observations

  def custom_pad(self, batch_observations):
    if self.use_disk_embeddings:
      seqs = [torch.tensor(x[0].embeddings, device=self.args['device']) for x in batch_observations]
    else:
      seqs = [torch.tensor(x[0].sentence, device=self.args['device']) for x in batch_observations]
    lengths = torch.tensor([len(x) for x in seqs], device=self.args['device'])
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    label_shape = batch_observations[0][1].shape
    maxlen = int(max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=self.args['device']) for x in seqs]
    for index, x in enumerate(batch_observations):
      length = x[1].shape[0]
      if len(label_shape) == 1:
        labels[index][:length] = x[1]
      elif len(label_shape) == 2:
        labels[index][:length,:length] = x[1]
      else:
        raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch_observations


class ELMoDataset(SimpleDataset):
  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    layer_index = self.args['model']['model_layer']
    print('Loading ELMo Pretrained Embeddings from {}; using layer {}'.format(pretrained_embeddings_path, layer_index))
    embeddings = self.generate_token_embeddings_from_hdf5(self.args, observations, pretrained_embeddings_path, layer_index)
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class SubwordDataset(SimpleDataset):
  @staticmethod
  def match_tokenized_to_untokenized(tokenized_sent, untokenized_sent):
    mapping = defaultdict(list)
    untokenized_sent_index = 0
    tokenized_sent_index = 1
    while (untokenized_sent_index < len(untokenized_sent) and
        tokenized_sent_index < len(tokenized_sent)):
      while (tokenized_sent_index + 1 < len(tokenized_sent) and
          tokenized_sent[tokenized_sent_index + 1].startswith('##')):
        mapping[untokenized_sent_index].append(tokenized_sent_index)
        tokenized_sent_index += 1
      mapping[untokenized_sent_index].append(tokenized_sent_index)
      untokenized_sent_index += 1
      tokenized_sent_index += 1
    return mapping

  def generate_subword_embeddings_from_hdf5(self, observations, filepath, elmo_layer, subword_tokenizer=None):
    raise NotImplementedError("Instead of making a SubwordDataset, make one of the implementing classes")


class BERTDataset(SubwordDataset):
  def generate_subword_embeddings_from_hdf5(self, observations, filepath, layer_index):
    """Load per-sentence BERT embeddings from HDF5.

    Supports two on-disk layouts:
      * 3D (n_stored_layers, n_words, hidden_dim): produced by the
        multi-layer generator. The HDF5 stores attrs['layer_indices']
        recording which absolute mBERT layers are present along axis 0.
        We translate the requested model.model_layer to the correct
        slot. If the requested layer was not saved, we raise.
      * 2D (n_words, hidden_dim): legacy single-layer dumps. The
        layer_index argument is ignored with a warning if it's nonzero.

    Layer 0 is the input embeddings; layers 1..L are transformer block
    outputs (L=12 for mBERT-base).
    """
    print(f'Loading BERT embeddings from {filepath} (layer={layer_index})...')
    out = []
    with h5py.File(filepath, 'r') as hf:
      stored_layers = list(hf.attrs.get('layer_indices', []))
      if stored_layers:
        if layer_index not in stored_layers:
          raise ValueError(
            f'model_layer={layer_index} was not saved in {filepath}. '
            f'Stored layers: {stored_layers}. Re-run generate_embeddings '
            f'with --layers all (or include this layer explicitly).')
        slot = stored_layers.index(layer_index)
      else:
        slot = layer_index

      sorted_indices = sorted(int(x) for x in hf.keys())
      warned_legacy = False
      for sent_idx in tqdm(sorted_indices, desc='[loading hdf5]'):
        feats = np.array(hf[str(sent_idx)])
        if feats.ndim == 3:
          if not (0 <= slot < feats.shape[0]):
            raise ValueError(
              f'Slot {slot} (for layer={layer_index}) out of range; '
              f'HDF5 has {feats.shape[0]} layers in {filepath}.')
          out.append(feats[slot])
        elif feats.ndim == 2:
          if layer_index != 0 and not warned_legacy:
            print(f'WARNING: legacy 2D HDF5 in {filepath}; '
                  f'ignoring model_layer={layer_index} and using the '
                  f'single stored layer.')
            warned_legacy = True
          out.append(feats)
        else:
          raise ValueError(
            f'Unexpected embedding ndim={feats.ndim} for sentence '
            f'{sent_idx} in {filepath}.')
    return out

  def optionally_add_embeddings(self, observations, pretrained_embeddings_path):
    layer_index = self.args['model']['model_layer']
    embeddings = self.generate_subword_embeddings_from_hdf5(
      observations, pretrained_embeddings_path, layer_index)
    if len(embeddings) != len(observations):
      raise ValueError(
        f'Alignment mismatch: {len(observations)} observations from '
        f'CoNLL-U vs {len(embeddings)} embedded sentences in '
        f'{pretrained_embeddings_path}. Did you regenerate the .txt and '
        f'.hdf5 from the same .conllu?')
    for i, (obs, emb) in enumerate(zip(observations, embeddings)):
      if emb.shape[0] != len(obs.sentence):
        raise ValueError(
          f'Sentence {i}: {len(obs.sentence)} CoNLL-U words but the '
          f'embedding has {emb.shape[0]} rows. Most likely the text '
          f'file was tokenized differently than the CoNLL-U FORM '
          f'column, or mBERT truncated a long sentence.')
    observations = self.add_embeddings_to_observations(observations, embeddings)
    return observations


class ObservationIterator(Dataset):
  """List Container for lists of Observations and labels for them."""

  def __init__(self, observations, task):
    self.observations = observations
    self.set_labels(observations, task)

  def set_labels(self, observations, task):
    self.labels = []
    for observation in tqdm(observations, desc='[computing labels]'):
      self.labels.append(task.labels(observation))

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    return self.observations[idx], self.labels[idx]
