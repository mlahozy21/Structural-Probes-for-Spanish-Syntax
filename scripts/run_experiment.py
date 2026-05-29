"""Loads configuration yaml and runs an experiment."""
from argparse import ArgumentParser
import os
from datetime import datetime
import shutil
import yaml
from tqdm import tqdm
import torch
import numpy as np
from scripts import model, probe, regimen, reporter, task, loss, data


def choose_task_classes(args):
  if args['probe']['task_name'] == 'parse-distance':
    task_class = task.ParseDistanceTask
    reporter_class = reporter.WordPairReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DistanceLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  elif args['probe']['task_name'] == 'parse-depth':
    task_class = task.ParseDepthTask
    reporter_class = reporter.WordReporter
    if args['probe_training']['loss'] == 'L1':
      loss_class = loss.L1DepthLoss
    else:
      raise ValueError("Unknown loss type for given probe type: {}".format(
        args['probe_training']['loss']))
  else:
    raise ValueError("Unknown probing task type: {}".format(
      args['probe']['task_name']))
  return task_class, reporter_class, loss_class


def choose_dataset_class(args):
  if args['model']['model_type'] in {'ELMo-disk', 'ELMo-random-projection', 'ELMo-decay'}:
    dataset_class = data.ELMoDataset
  elif args['model']['model_type'] == 'BERT-disk':
    dataset_class = data.BERTDataset
  else:
    raise ValueError("Unknown model type for datasets: {}".format(
      args['model']['model_type']))
  return dataset_class


def choose_probe_class(args):
  is_isometric = args['probe'].get('isometric', False)
  if args['probe']['task_signature'] == 'word':
    if is_isometric:
      return probe.IsometricOneWordPSDProbe
    elif args['probe']['psd_parameters']:
      return probe.OneWordPSDProbe
    else:
      return probe.OneWordNonPSDProbe
  elif args['probe']['task_signature'] == 'word_pair':
    if is_isometric:
      return probe.IsometricTwoWordPSDProbe
    elif args['probe']['psd_parameters']:
      return probe.TwoWordPSDProbe
    else:
      return probe.TwoWordNonPSDProbe
  else:
    raise ValueError("Unknown probe type (probe function signature): {}".format(
      args['probe']['task_signature']))


def choose_model_class(args):
  if args['model']['model_type'] == 'ELMo-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'BERT-disk':
    return model.DiskModel
  elif args['model']['model_type'] == 'ELMo-random-projection':
    return model.ProjectionModel
  elif args['model']['model_type'] == 'ELMo-decay':
    return model.DecayModel
  elif args['model']['model_type'] == 'pytorch_model':
    raise ValueError("Using pytorch models for embeddings not yet supported...")
  else:
    raise ValueError("Unknown model type: {}".format(
      args['model']['model_type']))


def run_train_probe(args, probe, dataset, model, loss, reporter, regimen):
  regimen.train_until_convergence(probe, model, loss,
      dataset.get_train_dataloader(), dataset.get_dev_dataloader())


def run_report_results(args, probe, dataset, model, loss, reporter, regimen):
  """Reports results from a trained probe.

  Always evaluates on the dev set. Optionally also evaluates on the test
  set when ``reporting.evaluate_test`` is true in the YAML config (default
  false: the test split should be touched only when reporting final
  numbers).
  """
  probe_params_path = os.path.join(args['reporting']['root'],
                                   args['probe']['params_path'])
  probe.load_state_dict(torch.load(probe_params_path,
                                   map_location=args['device'],
                                   weights_only=True))
  probe.eval()

  dev_dataloader = dataset.get_dev_dataloader()
  dev_predictions = regimen.predict(probe, model, dev_dataloader)
  reporter(dev_predictions, dev_dataloader, 'dev')

  if args['reporting'].get('evaluate_test', False):
    print('reporting.evaluate_test=true; evaluating on the held-out test set.')
    test_dataloader = dataset.get_test_dataloader()
    test_predictions = regimen.predict(probe, model, test_dataloader)
    reporter(test_predictions, test_dataloader, 'test')


def execute_experiment(args, train_probe, report_results):
  dataset_class = choose_dataset_class(args)
  task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)
  regimen_class = regimen.ProbeRegimen

  task = task_class()
  expt_dataset = dataset_class(args, task)
  expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
  expt_regimen = regimen_class(args)
  expt_loss = loss_class(args)

  if train_probe:
    print('Training probe...')
    run_train_probe(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)
  if report_results:
    print('Reporting results of trained probe...')
    run_report_results(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)


def setup_new_experiment_dir(args, yaml_args, reuse_results_path):
  now = datetime.now()
  date_suffix = '-'.join((str(x) for x in [now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond]))
  model_suffix = '-'.join((yaml_args['model']['model_type'], yaml_args['probe']['task_name']))

  if yaml_args['probe'].get('isometric', False):
    tqdm.write('Isometric probe detected. Saving results to subdirectory "iso".')
    yaml_args['reporting']['root'] = os.path.join(yaml_args['reporting']['root'], 'iso')

  if reuse_results_path:
    new_root = reuse_results_path
    tqdm.write('Reusing old results directory at {}'.format(new_root))
    if args.train_probe == -1:
      args.train_probe = 0
      tqdm.write('Setting train_probe to 0 to avoid squashing old params; '
          'explicitly set to 1 to override.')
  else:
    new_root = os.path.join(yaml_args['reporting']['root'], model_suffix + '-' + date_suffix +'/')
    tqdm.write('Constructing new results directory at {}'.format(new_root))

  yaml_args['reporting']['root'] = new_root
  os.makedirs(new_root, exist_ok=True)
  try:
    shutil.copyfile(args.experiment_config, os.path.join(yaml_args['reporting']['root'],
      os.path.basename(args.experiment_config)))
  except shutil.SameFileError:
    tqdm.write('Note, the config being used is the same as that already present in the results dir')


if __name__ == '__main__':
  argp = ArgumentParser()
  argp.add_argument('experiment_config')
  argp.add_argument('--results-dir', default='',
      help='Set to reuse an old results dir; if left empty, new directory is created')
  argp.add_argument('--train-probe', default=-1, type=int,
      help='Set to train a new probe.')
  argp.add_argument('--report-results', default=1, type=int,
      help='Set to report results; (optionally after training a new probe)')
  argp.add_argument('--seed', default=0, type=int,
      help='sets all random seeds for (within-machine) reproducibility')
  cli_args = argp.parse_args()
  # NOTE: seeding is unconditional. The previous `if cli_args.seed:` check
  # silently skipped seeding when seed=0 (since 0 is falsy in Python),
  # which made seed=0 runs non-reproducible.
  np.random.seed(cli_args.seed)
  torch.manual_seed(cli_args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  yaml_args = yaml.safe_load(open(cli_args.experiment_config, encoding='utf-8'))
  setup_new_experiment_dir(cli_args, yaml_args, cli_args.results_dir)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  yaml_args['device'] = device
  execute_experiment(yaml_args, train_probe=cli_args.train_probe, report_results=cli_args.report_results)
