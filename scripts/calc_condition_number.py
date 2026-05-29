"""Compute the condition number (geometric distortion) of a trained probe.

For the linear probe the projection matrix B is stored directly under the
key 'proj' in the state dict, so SVD on it gives the actual singular
spectrum used at inference.

For the isometric probe the projection lives behind a
``torch.nn.utils.parametrize.orthogonal`` wrapper. The pre-parametrization
tensor ('proj_layer.parametrizations.weight.original') is unconstrained,
so its singular values are NOT informative about the matrix the probe
actually uses. To get the real matrix we instantiate the probe class,
load_state_dict it, and read ``probe.proj_layer.weight``, which evaluates
the parametrization. By construction this matrix is orthogonal (its
singular values are all 1.0), so the condition number is 1.0 -- the
script confirms this and warns if it's not.

Usage
-----
  python -m scripts.calc_condition_number <path_to_predictor.params> \
      [--config <yaml>] [--isometric]

If --config is given, the script reads model.hidden_dim and
probe.maximum_rank from the YAML so it can rebuild the right probe class.
If not, the linear-probe path is used (it works for any state dict that
has a 'proj' key).
"""
from __future__ import annotations

import argparse
import sys

import torch
import yaml

from scripts import probe as probe_module


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('params_path')
  p.add_argument('--config', default=None,
                 help='YAML config used at training time (needed for the '
                      'isometric path).')
  p.add_argument('--isometric', action='store_true',
                 help='Force the isometric reconstruction path. Implied '
                      'when the state dict has no "proj" key.')
  return p.parse_args(argv)


def _build_isometric_probe(state_dict, config_path):
  if config_path is None:
    raise SystemExit(
      'Isometric probe detected but no --config was given. Re-run with '
      '--config <yaml> so the probe class can be instantiated with the '
      'right rank and hidden_dim.')
  cfg = yaml.safe_load(open(config_path, encoding='utf-8'))
  cfg['device'] = torch.device('cpu')
  if cfg['probe']['task_signature'] == 'word_pair':
    cls = probe_module.IsometricTwoWordPSDProbe
  else:
    cls = probe_module.IsometricOneWordPSDProbe
  probe = cls(cfg)
  probe.load_state_dict(state_dict)
  probe.eval()
  return probe.proj_layer.weight.detach().cpu()


def get_projection_matrix(params_path, config_path, force_isometric):
  state_dict = torch.load(params_path, map_location='cpu', weights_only=False)
  if 'proj' in state_dict and not force_isometric:
    return state_dict['proj'], 'linear'
  return _build_isometric_probe(state_dict, config_path), 'isometric'


def main(argv=None):
  args = parse_args(argv)
  print(f'Loading probe from: {args.params_path}')
  matrix, kind = get_projection_matrix(args.params_path, args.config,
                                       args.isometric)

  _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
  max_scale = S.max().item()
  min_scale = S.min().item()
  cond_number = max_scale / (min_scale + 1e-9)

  print('-' * 40)
  print(f'Probe kind:     {kind}')
  print(f'Matrix shape:   {tuple(matrix.shape)}')
  print(f'Sigma_max:      {max_scale:.4f}')
  print(f'Sigma_min:      {min_scale:.4f}')
  print(f'Sigma_mean:     {S.mean().item():.4f}')
  print(f'Sigma_median:   {S.median().item():.4f}')
  print('-' * 40)
  print(f'CONDITION NUMBER (sigma_max / sigma_min): {cond_number:.4f}')
  print('-' * 40)

  if kind == 'isometric':
    if abs(cond_number - 1.0) > 1e-3:
      print('WARNING: isometric probe should have kappa ~= 1.0. Got '
            f'{cond_number:.4f}. Check the parametrization.')
    else:
      print('OK: orthogonality constraint satisfied.')
  else:
    if cond_number < 1.5:
      print('Interpretation: minimal distortion; the syntactic subspace '
            'is nearly isometric.')
    elif cond_number < 5.0:
      print('Interpretation: moderate anisotropy; some directions are '
            'amplified more than others to recover syntax.')
    else:
      print('Interpretation: high anisotropy. The syntactic signal lives '
            'along low-variance directions of the embedding space and '
            'the probe must amplify them substantially to expose the '
            'parse-tree geometry.')


if __name__ == '__main__':
  main(sys.argv[1:])
