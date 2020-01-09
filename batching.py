"""Produce more or less evenly distributed batches from variable length data."""
import collections
import random

import numpy as np
import torch

import tokenizer


def produce_batches(data, batch_size):
  dataset = TokenDataset(data)
  data_loader = torch.utils.data.DataLoader(
      dataset,
      batch_sampler=BucketSampler(dataset, batch_size=batch_size),
      collate_fn=PadBatch(dim=0))
  return data_loader


class TokenDataset(torch.utils.data.Dataset):
  """The dataset that holds the corpus of data as tensors."""
  def __init__(self, iterable):
    self._inputs = []
    self._targets = []

    for sample in iterable:
      self._inputs.append(_to_tensor(sample['inputs']))
      self._targets.append(_to_tensor(sample['targets']))

    assert len(self._inputs) == len(self._targets)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    return self._inputs[idx], self._targets[idx]

  def __len__(self):
    return len(self._inputs)


def _to_tensor(line):
  """Convert a string with space-separated numbers into a tensor."""
  return torch.from_numpy(np.asarray([int(x) for x in line.split()]))


class BucketSampler(torch.utils.data.Sampler):
  """Arranges batches such the total length is constant."""

  def __init__(self, dataset, batch_size, maximum_length=None):
    super(BucketSampler, self).__init__(dataset)
    if maximum_length is None:
      maximum_length = batch_size
    # Create buckets for sequence lengths.  Each bucket has a minimum and
    # a maximum boundary up to maximum_length.
    buckets_min, buckets_max = _create_min_max_boundaries(maximum_length)

    # Bucket the sequences by length.  We store indices not the sequences
    # themselves.
    buckets = collections.defaultdict(list)
    for index, (inputs, targets) in enumerate(dataset):
      length = max(inputs.shape[0], targets.shape[0])
      if length > maximum_length:
        continue

      bucket_map = torch.le(buckets_min, length) & torch.ge(buckets_max, length)
      bucket_id = torch.min(torch.nonzero(bucket_map)).item()
      buckets[bucket_id].append(index)

    # A batch is going to have a lot of small sequences or a few big sequences
    # in it, so that the total length of the batch is maximum_length.
    bucket_batch_sizes = [batch_size // x for x in buckets_max]

    # Create batches of indices of the right size.
    self._batches_of_indices = []
    for bucket_id, indices in buckets.items():
      indices = torch.IntTensor(indices)
      _shuffle(indices)
      self._batches_of_indices += torch.chunk(indices,
                                              bucket_batch_sizes[bucket_id])
    random.shuffle(self._batches_of_indices)

  def __iter__(self):
    for batch_of_indices in self._batches_of_indices:
      yield batch_of_indices

  MIN_BOUNDARY = 8
  BOUNDARY_SCALE = 1.1


def _create_min_max_boundaries(max_length,
                               min_boundary=BucketSampler.MIN_BOUNDARY,
                               boundary_scale=BucketSampler.BOUNDARY_SCALE):
  bucket_boundaries = []
  boundary = min_boundary

  while boundary < max_length:
    bucket_boundaries.append(boundary)
    boundary = max(boundary + 1, int(boundary * boundary_scale))

  buckets_min = [0] + bucket_boundaries
  buckets_max = bucket_boundaries + [max_length]
  return (torch.from_numpy(np.asarray(buckets_min)),
          torch.from_numpy(np.asarray(buckets_max)))


def _shuffle(vector):
  vector[torch.arange(vector.numel())] = vector[torch.randperm(vector.numel())]


class PadBatch:
  """After the batch is formed, pad its constituents as needed."""

  def __init__(self, dim):
    self._dim = dim

  _PAD_ID = tokenizer.Tokenizer.RESERVED_TOKENS.index(tokenizer.Tokenizer.PAD)

  def __call__(self, batch):
    return self._collate(batch)

  def _collate(self, batch):
    max_inputs_len = max(map(lambda s: len(s[0]), batch))
    max_targets_len = max(map(lambda s: len(s[1]), batch))

    def pad(inputs, targets):
      return (self._pad_tensor(inputs, pad=max_inputs_len),
              self._pad_tensor(targets, pad=max_targets_len))

    batch = list(map(lambda sample: pad(*sample), batch))

    return list(map(torch.stack, zip(*batch)))

  def _pad_tensor(self, vec, pad):
    pad_size = list(vec.shape)
    pad_size[self._dim] = pad - vec.shape[self._dim]
    pad_tensor = torch.full(pad_size, PadBatch._PAD_ID, dtype=torch.long)
    return torch.cat([vec, pad_tensor], dim=self._dim)
