"""Train one probe per layer and aggregate the metrics into a CSV.

This produces the canonical "Spearman vs. layer index" curve that
characterises *where* in the network syntactic information lives. Hewitt
& Manning (2019) found this peaks around layers 7-8 in BERT-base; our
hypothesis with mBERT and Spanish AnCora is similar.

Usage
-----
  python -m scripts.run_layer_sweep es_ancora.yaml --layers 0 7 8 12 \
       --seed 0

The output CSV has columns:
  layer, dev_spearman_5_50, dev_uuas, test_spearman_5_50, test_uuas
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
from datetime import datetime

import numpy as np
import torch
import yaml

from scripts.run_experiment import execute_experiment


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('experiment_config')
  p.add_argument('--layers', nargs='+', type=int,
                 default=list(range(13)),
                 help='Layer indices to probe. Default: all 13 mBERT layers.')
  p.add_argument('--seed', type=int, default=0)
  return p.parse_args(argv)


def _read_metric(path):
  if not os.path.exists(path):
    return None
  with open(path, encoding='utf-8') as f:
    line = f.readline().strip()
  try:
    return float(line.split('\t')[0])
  except (ValueError, IndexError):
    return None


def main(argv=None):
  args = parse_args(argv)
  base_yaml = yaml.safe_load(
    open(args.experiment_config, encoding='utf-8'))
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  base_root = base_yaml['reporting']['root']
  stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  parent_dir = os.path.join(base_root, f'layersweep-{stamp}')
  os.makedirs(parent_dir, exist_ok=True)

  rows = []
  for layer in args.layers:
    print('\n' + '=' * 60)
    print(f'  Probing layer {layer}')
    print('=' * 60)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    yaml_args = copy.deepcopy(base_yaml)
    yaml_args['device'] = device
    yaml_args['model']['model_layer'] = layer
    yaml_args['reporting']['root'] = os.path.join(parent_dir, f'layer{layer:02d}')
    os.makedirs(yaml_args['reporting']['root'], exist_ok=True)

    execute_experiment(yaml_args, train_probe=True, report_results=True)

    rows.append({
      'layer': layer,
      'dev_spearman_5_50': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'dev.spearmanr-5_50-mean')),
      'dev_uuas': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'dev.uuas')),
      'test_spearman_5_50': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'test.spearmanr-5_50-mean')),
      'test_uuas': _read_metric(os.path.join(
        yaml_args['reporting']['root'], 'test.uuas')),
    })

  csv_path = os.path.join(parent_dir, 'sweep.csv')
  with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(
      f, fieldnames=['layer', 'dev_spearman_5_50', 'dev_uuas',
                     'test_spearman_5_50', 'test_uuas'])
    writer.writeheader()
    writer.writerows(rows)
  print('\nWrote', csv_path)
  for r in rows:
    print(r)


if __name__ == '__main__':
  main()
