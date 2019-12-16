"""Download and store datasets."""

import collections
import io
import itertools
import pathlib
import os
import logging
import tarfile
import urllib.request

import tokenizer as tokenizr


_LOGGER = logging.getLogger(__name__)


def obtain(dataset_directory):
  for dataset in itertools.chain(*_DATASETS_.values()):
    download_if_not_there(dataset, dataset_directory)

  tokenizer = create_tokenizer(_DATASETS_['train'], dataset_directory)

  for label, datasets in _DATASETS_.items():
    full_corpus = read_corpus(datasets, dataset_directory)
    encode_corpus(full_corpus, tokenizer, dataset_directory, prefix=label)


Dataset = collections.namedtuple(
    'Dataset', ('url', 'language_1_filename', 'language_2_filename'))


_DATASETS_ = {
    'train': (
        Dataset(url='http://data.statmt.org/wmt18/translation-task/' +
                'training-parallel-nc-v13.tgz',
                language_1_filename='training-parallel-nc-v13/' +
                'news-commentary-v13.de-en.en',
                language_2_filename='training-parallel-nc-v13/' +
                'news-commentary-v13.de-en.de'),
        Dataset(
            url='http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
            language_1_filename='commoncrawl.de-en.en',
            language_2_filename='commoncrawl.de-en.de'),
        Dataset(
            url='http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
            language_1_filename='training/europarl-v7.de-en.en',
            language_2_filename='training/europarl-v7.de-en.de'),
    ),
    'eval':
    (Dataset(url='http://data.statmt.org/wmt17/translation-task/dev.tgz',
             language_1_filename='dev/newstest2013.en',
             language_2_filename='dev/newstest2013.de'), ),
}

def download_if_not_there(dataset, local_directory):
  def does_exist(filename):
    return (pathlib.Path(local_directory) / filename).exists()

  if not (does_exist(dataset.language_1_filename) and
          does_exist(dataset.language_2_filename)):
    _LOGGER.info('Downloading %s', dataset.url)
    download_and_extract(dataset, local_directory)


def download_and_extract(dataset, destination_directory):
  """Download a tar archive for the dataset and extract it to the local disk.

  Args:
    dataset:  The dataset that is going to be downloaded.  Only
      files that match `dataset.language_pair` are going to be stored.
    destination_directory:  The directory where the archive is going to be
      extracted. For example, `file1` inside the archive at
      `http://example.com/archive.tar.gz` is going to be extracted to
      `destination_directory/archive/file1`.
  """
  with urllib.request.urlopen(dataset.url) as response:
    download = response.read()

  in_memory_file = io.BytesIO(download)
  archive = tarfile.open(fileobj=in_memory_file)

  def needed_files_only(tarinfo):
    return tarinfo.name in (dataset.language_1_filename,
                            dataset.language_2_filename)

  for untarred_file in filter(needed_files_only, archive.getmembers()):
    archive.extract(untarred_file, path=destination_directory)


def create_tokenizer(datasets, directory):
  """Builds a new vocabulary if needed and returns a Tokenizer."""
  vocabulary_file = pathlib.Path(directory) / tokenizr.Tokenizer.VOCAB_FILENAME
  if vocabulary_file.exists():
    _LOGGER.info('Re-using the vocabulary at %s', vocabulary_file)
    return tokenizr.Builder.from_file(vocabulary_file)

  samples = sample_corpus(datasets, directory,
                          sample_within_byte_budget=1e6)
  tokenizer = tokenizr.Builder(2**15).from_corpus(samples)
  tokenizer.store_to_file(vocabulary_file)
  return tokenizer


def sample_corpus(datasets, directory, sample_within_byte_budget=1e6):
  """Sample the datasets.

  Args:
      datasets: Dataset objects that describe what to read.
      directory:  The local directory where the datasets are stored.
      sample_within_byte_budget:  Read this many bytes by sampling
        across each file.

  Returns:
      Text lines are returned one by one.
  """
  def read_lines(filename):
    _LOGGER.info('Sampling from %s', filename)
    with open(filename, 'r') as opened_file:
      counter = 0
      remaining_byte_budget = sample_within_byte_budget
      file_size = os.path.getsize(filename)
      for line in opened_file:
        if counter < int(file_size / sample_within_byte_budget / 2):
          counter += 1
        else:
          if remaining_byte_budget <= 0:
            break
          line = line.strip()
          remaining_byte_budget -= len(line)
          counter = 0
          yield line

  for dataset in datasets:
    local_path = pathlib.Path(directory)
    language_1_path = local_path / dataset.language_1_filename
    language_2_path = local_path / dataset.language_2_filename

    for path in [language_1_path, language_2_path]:
      for line in read_lines(path):
        yield line


def encode_corpus(full_corpus, tokenizer, dataset_directory, prefix=None):
  """Encode the whole corpus and store it locally.

  Args:
      full_corpus:  Iterable of dict{inputs,targets} that represent the full
          corpus.
      tokenizer:  Tokenizer instance that is going encode each sample.
      dataset_directory:  The local path where the encoded files are going to
          stored.  There is going to be a separate file for inputs and targets.
          If the encoded files are already present then the function is not
          going to repeat the process.
      prefix:  An optional prefix that can be appeneded to the filenames.
  """
  dataset_directory = pathlib.Path(dataset_directory)

  prefix = prefix + '-'
  inputs_file_path = dataset_directory / (prefix + 'inputs')
  targets_file_path = dataset_directory / (prefix + 'targets')

  if (inputs_file_path.exists() and targets_file_path.exists()):
    _LOGGER.info('Re-using encoded dataset files in %s: %s and %s.',
                 dataset_directory, inputs_file_path, targets_file_path)
    return

  eos_id = tokenizer.RESERVED_TOKENS.index(tokenizer.EOS)

  with open(inputs_file_path, 'w') as inputs_file, \
       open(targets_file_path, 'w') as targets_file:
    samples_written = 0
    for sample in full_corpus:
      inputs = tokenizer.encode(sample['inputs']) + [eos_id]
      targets = tokenizer.encode(sample['targets']) + [eos_id]

      inputs_file.write(' '.join([str(i) for i in inputs]) + ' ')
      targets_file.write(' '.join([str(t) for t in targets]) + ' ')
      samples_written += 1
      if samples_written % 100000 == 0:
        _LOGGER.info('Encoded %d samples so far...', samples_written)
    _LOGGER.info('Encoded %d samples total.', samples_written)


def read_corpus(datasets, directory):
  """Read the datasets.

  Args:
      datasets: Dataset objects that describe what to read.
      directory:  The local directory where the datasets are stored.

  Returns:
      A dictionary with input and targets is going to be returned.
  """
  def read_lines(filename):
    _LOGGER.info('Reading from %s', filename)
    with open(filename, 'r') as opened_file:
      for line in opened_file:
        yield line.strip()

  for dataset in datasets:
    local_path = pathlib.Path(directory)
    language_1_path = local_path / dataset.language_1_filename
    language_2_path = local_path / dataset.language_2_filename

    for inputs, targets in zip(read_lines(language_1_path),
                               read_lines(language_2_path)):
      yield {'inputs': inputs, 'targets': targets}
