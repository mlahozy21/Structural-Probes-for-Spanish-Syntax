"""Run the same probe configuration with multiple random seeds.

Each seed produces its own results subdirectory. After all runs we read
the per-seed dev (and optionally test) Spearman/UUAS files and write a
single ``aggregate.json`` with mean and standard deviation, plus the raw
per-seed values, into the parent reporting directory.

This is the chunk of rigor most often missing in single-shot probing
studies (Hewitt & Liang, 2019). Reporting std across >=3 seeds is a hard
requirement for any *NACL/EMNLP/ACL claim about probe selectivity or
about a relative gap between two probes (e.g. linear vs isometric).

Usage
-----
  python -m scripts.run_multiseed es_ancora.yaml --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from datetime import datetime

import numpy as np
import torch
import yaml

from scripts.run_experiment import execute_experiment


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('experiment_config')
  p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
  p.add_argument('--report-results', type=int, default=1)
  p.add_argument('--train-probe', type=int, default=-1)
  return p.parse_args(argv)


def _read_metric(path: str):
  """Reads a single-float metric file (e.g. dev.uuas) or returns None."""
  if not os.path.exists(path):
    return None
  with open(path, encoding='utf-8') as f:
    line = f.readline().strip()
  try:
    return float(line.split('\t')[0])
  except (ValueError, IndexError):
    return None


def _agg(values):
  """Mean and SAMPLE stdev (Bessel correction) over non-None values.

  Sample stdev (n-1 denominator) is what people normally report for a
  set of training runs -- population stdev would underestimate the true
  spread when n is small (typical 3-5 seeds).
  """
  clean = [v for v in values if v is not None]
  if not clean:
    return None
  if len(clean) == 1:
    return {'mean': clean[0], 'std': 0.0, 'n': 1, 'values': clean,
            'note': 'std undefined for n=1'}
  return {
    'mean': statistics.mean(clean),
    'std': statistics.stdev(clean),
    'n': len(clean),
    'values': clean,
  }


def main(argv=None):
  args = parse_args(argv)
  base_yaml = yaml.safe_load(
    open(args.experiment_config, encoding='utf-8'))
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  base_root = base_yaml['reporting']['root']
  stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  parent_dir = os.path.join(base_root, f'multiseed-{stamp}')
  os.makedirs(parent_dir, exist_ok=True)

  per_seed = {}
  for seed in args.seeds:
    print('\n' + '=' * 60)
    print(f'  Running seed {seed}')
    print('=' * 60)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    yaml_args = copy.deepcopy(base_yaml)
    yaml_args['device'] = device
    yaml_args['reporting']['root'] = os.path.join(parent_dir, f'seed{seed}')
    os.makedirs(yaml_args['reporting']['root'], exist_ok=True)

    execute_experiment(yaml_args,
                       train_probe=(args.train_probe == 1
                                    or args.train_probe == -1),
                       report_results=bool(args.report_results))

    per_seed[seed] = {
      'dev_uuas': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'dev.uuas')),
      'dev_spearman_mean_5_50': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'dev.spearmanr-5_50-mean')),
      'test_uuas': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'test.uuas')),
      'test_spearman_mean_5_50': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'test.spearmanr-5_50-mean')),
    }

  aggregate = {
    'config_path': args.experiment_config,
    'seeds': args.seeds,
    'per_seed': per_seed,
    'dev_uuas': _agg([per_seed[s]['dev_uuas'] for s in args.seeds]),
    'dev_spearman_mean_5_50': _agg(
      [per_seed[s]['dev_spearman_mean_5_50'] for s in args.seeds]),
    'test_uuas': _agg([per_seed[s]['test_uuas'] for s in args.seeds]),
    'test_spearman_mean_5_50': _agg(
      [per_seed[s]['test_spearman_mean_5_50'] for s in args.seeds]),
  }

  out_path = os.path.join(parent_dir, 'aggregate.json')
  with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(aggregate, f, indent=2)
  print('\nAggregate metrics written to', out_path)
  print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
  main()
