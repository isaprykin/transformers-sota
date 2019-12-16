"""Download datasets and train transformer-based models."""
import argparse
import logging
import datasets


def main():
  """The main entry point to transformers-sota."""

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_directory',
                      default='/tmp/translation_datasets/',
                      help='The local directory where the datasets are stored.')
  parser.add_argument('--log_level', default='WARNING',
                      help='The desired logging level.')
  args = parser.parse_args()
  logging.basicConfig(level=getattr(logging, args.log_level.upper(), None))

  datasets.obtain(args.dataset_directory)


if __name__ == '__main__':
  main()
