# -*- coding: utf-8 -*-
from typing import Iterable, Tuple, Union, Optional, Type

import torch

from ttab.scenarios import Scenario
from ttab.api import PyTorchDataset, Batch
from ttab.loads.datasets.datasets import WrapperDataset


D = Union[torch.utils.data.Dataset, PyTorchDataset]


class BaseLoader(object):
    def __init__(self, dataset: PyTorchDataset):
        self.dataset = dataset

    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        generator: Optional[torch.Generator] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        yield from self.dataset.iterator(
            batch_size,
            shuffle,
            repeat,
            ref_num_data,
            num_workers,
            sampler,
            generator,
            pin_memory,
            drop_last,
        )


class NormalLoader(object):
    """This class can be used for TTT variants (TODO)."""

    def __init__(self, dataset: PyTorchDataset):
        self.dataset = dataset

    def iterator(
        self,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = False,
        ref_num_data: Optional[int] = None,
        num_workers: int = 1,
        sampler: Optional[torch.utils.data.Sampler] = None,
        generator: Optional[torch.Generator] = None,
        pin_memory: bool = True,
        drop_last: bool = True,
    ) -> Iterable[Tuple[int, float, Batch]]:
        yield from self.dataset.iterator(
            batch_size,
            shuffle,
            repeat,
            ref_num_data,
            num_workers,
            sampler,
            generator,
            pin_memory,
            drop_last,
        )


scenario2loader = {
    "tent": BaseLoader,
    "no_adaptation": BaseLoader,
    "bn_adapt": BaseLoader,
    "memo": BaseLoader,
    "shot": BaseLoader,
    "t3a": BaseLoader,
    "ttt": NormalLoader,
    "ttt_plus_plus": NormalLoader,
    "note": BaseLoader,
    "sar": BaseLoader,
    "conjugate_pl": BaseLoader,
}


def _init_dataset(dataset: D, device: str) -> PyTorchDataset:
    if isinstance(dataset, torch.utils.data.Dataset):
        return WrapperDataset(dataset, device)
    else:
        return dataset


def get_test_loader(dataset: D, scenario: Scenario, device: str) -> Type[BaseLoader]:
    dataset: PyTorchDataset = _init_dataset(dataset, device)
    return scenario2loader[scenario.model_adaptation_method](dataset)


def get_auxiliary_loader(dataset: D, device: str) -> Type[BaseLoader]:
    dataset: PyTorchDataset = _init_dataset(dataset, device)
    return BaseLoader(dataset)
