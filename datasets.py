"""Download and store datasets."""

import collections
import io
import pathlib
import os
import logging
import tarfile
import urllib.request

import tokenizer


_LOGGER = logging.getLogger(__name__)


def obtain(dataset_directory):
  for dataset in _DATASETS_:
    download_if_not_there(dataset, dataset_directory)

  samples = sample_corpus(_DATASETS_, dataset_directory,
                          sample_within_byte_budget=1e6)
  tokenizr = tokenizer.Builder(2**15).from_corpus(samples)
  tokenizr.store_to_file(pathlib.Path(dataset_directory) / 'vocab.subwords')


Dataset = collections.namedtuple(
    'Dataset', ('url', 'language_1_filename', 'language_2_filename'))


_DATASETS_ = (
    Dataset(url='http://data.statmt.org/wmt18/translation-task/' +
            'training-parallel-nc-v13.tgz',
            language_1_filename='training-parallel-nc-v13/' +
            'news-commentary-v13.de-en.en',
            language_2_filename='training-parallel-nc-v13/' +
            'news-commentary-v13.de-en.de'),
    Dataset(url='http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz',
            language_1_filename='commoncrawl.de-en.en',
            language_2_filename='commoncrawl.de-en.de'),
    Dataset(url='http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz',
            language_1_filename='training/europarl-v7.de-en.en',
            language_2_filename='training/europarl-v7.de-en.de'),
)


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
