"""Linguistic tasks that map an Observation to a label tensor.

Two tasks are implemented, both following Hewitt & Manning (2019):

  * ParseDistanceTask: pairwise tree distance d_T(w_i, w_j) for every pair
    of words. Returned as a (n, n) float matrix.
  * ParseDepthTask: depth in the dependency tree for each word. Returned
    as an (n,) float vector.

Notes
-----
The CoNLL-U files store HEAD as a 1-based index into the word list, with 0
denoting the special ROOT node. Contraction range rows (e.g. `13-14 al`)
are filtered out at load time in ``data.py``; by the time observations
reach this module the indices are already a contiguous 1..n numbering.
"""

import warnings

import numpy as np
import torch


def _parse_head_indices(observation):
  """Return a list of integer head indices, one per word.

  Robust to:
    * Heads stored as the string '_' (treated as ROOT, i.e. 0).
    * Heads that fail integer conversion (treated as ROOT and warned).
  """
  raw_heads = observation.head_indices if hasattr(
    observation, 'head_indices') else observation[6]

  heads = []
  for elt in raw_heads:
    if elt == '_':
      heads.append(0)
      continue
    try:
      heads.append(int(elt))
    except (ValueError, TypeError):
      warnings.warn(f'Unparseable HEAD value {elt!r}; treating as ROOT.')
      heads.append(0)
  return heads


class Task:
  """Abstract base. Subclasses turn an Observation into a label tensor."""

  @staticmethod
  def labels(observation):
    raise NotImplementedError


class ParseDistanceTask(Task):
  """Pairwise dependency tree distance d_T(w_i, w_j)."""

  @staticmethod
  def labels(observation):
    heads = _parse_head_indices(observation)
    seq_len = len(heads)

    # Initialise distance matrix with +inf, then 0 on the diagonal.
    dist = np.full((seq_len, seq_len), np.inf, dtype=np.float32)
    np.fill_diagonal(dist, 0.0)

    # Fill direct (parent <-> child) edges with weight 1. Trees are
    # undirected for distance purposes.
    for i, head in enumerate(heads):
      if head == 0:
        continue  # ROOT has no parent
      head_idx = head - 1  # 1-based -> 0-based
      if head_idx < 0 or head_idx >= seq_len:
        warnings.warn(
          f'HEAD index {head} out of range for sentence length {seq_len}; '
          f'edge skipped.')
        continue
      dist[i, head_idx] = 1.0
      dist[head_idx, i] = 1.0

    # Floyd-Warshall (vectorised over (i, j); the outer k loop is Python).
    # Trees have unique paths so this gives exact tree distance.
    for k in range(seq_len):
      dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    # If any entries remain +inf the dependency graph has a disconnected
    # component (annotation error or malformed sentence). We replace those
    # with a large finite value to avoid NaNs downstream, but warn so the
    # user can investigate.
    if not np.isfinite(dist).all():
      warnings.warn(
        'Disconnected component detected in dependency graph; '
        'masking with large finite value.')
      dist[~np.isfinite(dist)] = float(seq_len + 10)

    return torch.tensor(dist, dtype=torch.float)


class ParseDepthTask(Task):
  """Depth of each word in the dependency tree (root has depth 0)."""

  @staticmethod
  def labels(observation):
    heads = _parse_head_indices(observation)
    seq_len = len(heads)

    depths = torch.zeros(seq_len, dtype=torch.float)
    for i in range(seq_len):
      depth = 0
      current = i + 1  # 1-based
      safety = 0
      while current != 0:
        try:
          current = heads[current - 1]
        except IndexError:
          warnings.warn(
            f'HEAD chain out of bounds for word {i}; depth truncated.')
          break
        if current != 0:
          depth += 1
        safety += 1
        if safety > seq_len + 5:
          warnings.warn(
            f'Cycle detected while computing depth of word {i}; '
            f'depth truncated to {depth}.')
          break
      depths[i] = depth
    return depths
