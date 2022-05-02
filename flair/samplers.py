import logging
import random
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

log = logging.getLogger("flair")


class FlairSampler(Sampler):
    def set_dataset(self, data_source):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)

    def __len__(self):
        return self.num_samples


class ImbalancedClassificationDatasetSampler(FlairSampler):
    """Use this to upsample rare classes and downsample common classes in your unbalanced classification dataset."""

    def __init__(self):
        super(ImbalancedClassificationDatasetSampler, self).__init__(None)

    def set_dataset(self, data_source):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        self.indices = list(range(len(data_source)))

        # first determine the distribution of classes in the dataset
        label_count: Dict[str, int] = defaultdict(int)
        for sentence in data_source:
            for label in sentence.labels:
                label_count[label.value] += 1

        # weight for each sample
        offset = 0
        weights = [1.0 / (offset + label_count[data_source[idx].labels[0].value]) for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))


class ChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. This causes some order of the data to be preserved,
    while still shuffling the data.
    """

    def __init__(self, block_size=5, plus_window=5):
        super(ChunkSampler, self).__init__(None)
        self.block_size = block_size
        self.plus_window = plus_window
        self.data_source = None

    def __iter__(self):
        data = list(range(len(self.data_source)))

        blocksize = self.block_size + random.randint(0, self.plus_window)

        log.info(f"Chunk sampling with blocksize = {blocksize} ({self.block_size} + {self.plus_window})")

        # Create blocks
        blocks = [data[i : i + blocksize] for i in range(0, len(data), blocksize)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)


class ExpandingChunkSampler(FlairSampler):
    """Splits data into blocks and randomizes them before sampling. Block size grows with each epoch.
    This causes some order of the data to be preserved, while still shuffling the data.
    """

    def __init__(self, step=3):
        """Initialize by passing a block_size and a plus_window parameter.
        :param data_source: dataset to sample from
        """
        super(ExpandingChunkSampler, self).__init__(None)
        self.block_size = 1
        self.epoch_count = 0
        self.step = step

    def __iter__(self):
        self.epoch_count += 1

        data = list(range(len(self.data_source)))

        log.info(f"Chunk sampling with blocksize = {self.block_size}")

        # Create blocks
        blocks = [data[i : i + self.block_size] for i in range(0, len(data), self.block_size)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]

        if self.epoch_count % self.step == 0:
            self.block_size += 1

        return iter(data)

class AlphaSampler(FlairSampler):

    def __init__(self, alpha=1, sample_size=None, batch_corpus_together=False):
        self.alpha = alpha
        self.size = sample_size
        self.batch_corpus_together = batch_corpus_together

    def __iter__(self):
        relative_shares = [dataset_size / sum(self.sizes) for dataset_size in self.sizes]
        denominator = sum([relative_share ** self.alpha for relative_share in relative_shares])
        sample_sizes = [int((1 / pi) * ((pi ** self.alpha) / denominator) * size) for pi, size in zip(relative_shares, self.sizes)]
        sample_sizes = [int(x / sum(sample_sizes) * self.size) for x in sample_sizes]
        indices = []
        for task_weights, task_indices, sample_size in zip(self.weights, self.indices, sample_sizes):
            task_samples = random.choices(task_indices, weights=task_weights, k=sample_size)
            indices += task_samples
        if not self.batch_corpus_together:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.size

    def set_dataset(self, data_source):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        self.data_source = data_source
        self.sizes = np.diff(data_source.cumulative_sizes, prepend=0).tolist()
        self.weights, self.indices = self._get_weights()
        self.size = sum(self.sizes) if self.size is None else self.size

    def _get_weights(self):
        weights = []
        indices = []
        start = 0
        for size in self.sizes:
            end = start + size
            weights.append(torch.ones(size) / size)
            indices.append(list(np.arange(start, end)))
            start = start + size
        return weights, indices

class AlphaSamplerForTARS(FlairSampler):

    def __init__(self, alpha=1, sample_size=None, batch_corpus_together=False):
        self.alpha = alpha
        self.size = sample_size
        self.batch_corpus_together = batch_corpus_together

    def __iter__(self):
        relative_shares = [dataset_size / sum(self.sizes) for dataset_size in self.sizes]
        denominator = sum([relative_share ** self.alpha for relative_share in relative_shares])
        sample_sizes = [int((1 / pi) * ((pi ** self.alpha) / denominator) * size) for pi, size in zip(relative_shares, self.sizes)]
        sample_sizes = [int(x / sum(sample_sizes) * self.size) for x in sample_sizes]
        indices = []
        for task_weights, task_indices, sample_size in zip(self.weights, self.indices, sample_sizes):
            task_samples = random.choices(task_indices, weights=task_weights, k=sample_size)
            indices += task_samples
        if not self.batch_corpus_together:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.size

    def set_dataset(self, data_source):
        """
        Initialize by passing a classification dataset with labels, i.e. either TextClassificationDataSet or
        :param data_source:
        """
        self.data_source = data_source
        self.sizes = np.diff(data_source.cumulative_sizes, prepend=0).tolist()
        self.weights, self.indices = self._get_weights()
        self.size = sum(self.sizes) if self.size is None else self.size

    def _get_weights(self):
        _number_of_labels_per_sentence = []
        for sentence in self.data_source:
            _number_of_labels_per_sentence.append(
                len(set([label.value for label in sentence.get_labels("ner")])) + 1
            )
        weights = []
        indices = []
        start = 0
        for size in self.sizes:
            end = start + size
            weights.append(torch.tensor(_number_of_labels_per_sentence[start:end]) / size)
            indices.append(list(np.arange(start, end)))
            start = start + size
        return weights, indices

class KShotSampler(FlairSampler):

    def __init__(self, k: int, seed: int = None):
        self.k = k
        self.seed = seed

    def __iter__(self):
        batch = []
        for label, indices in self.label_to_idx.items():
            random.seed(self.seed)
            batch.append(random.sample(indices, self.k))
        batch = [item for sublist in batch for item in sublist]
        return iter(batch)

    def __len__(self):
        return self.num_samples * self.num_labels

    def set_dataset(self, data_source):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        label_to_idx = {}
        for i in range(self.num_samples):
            labels = set([label.value for label in self.data_source[i].get_labels("ner")])
            for label in labels:
                if label in label_to_idx:
                    label_to_idx[label].append(i)
                else:
                    label_to_idx[label] = [i]
        self.num_labels = len(label_to_idx)
        self.label_to_idx = label_to_idx
