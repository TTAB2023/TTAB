# -*- coding: utf-8 -*-
import random

from ttab.api import PyTorchDataset


class DatasetSampling(object):
    def __init__(self, test_domain) -> None:
        self.domain_sampling_name = test_domain.domain_sampling_name
        self.domain_sampling_value = test_domain.domain_sampling_value
        self.domain_sampling_ratio = test_domain.domain_sampling_ratio

    def sample(self, dataset: PyTorchDataset, random_seed=None) -> PyTorchDataset:
        if self.domain_sampling_name == "uniform":
            return self._uniform_sample(dataset=dataset, random_seed=random_seed)

    def _uniform_sample(self, dataset: PyTorchDataset, random_seed=None):
        """This function uniformly samples data from the original dataset without replacement."""
        random.seed(random_seed)
        sampled_list = random.sample(
            dataset.dataset.indices,
            int(self.domain_sampling_ratio * dataset.dataset.data_size),
        )
        sampled_list.sort()
        dataset.replace_indices(
            indices_pattern="new", new_indices=sampled_list, random_seed=random_seed
        )

        return dataset
