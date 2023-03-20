# -*- coding: utf-8 -*-
import os
import functools
from typing import Callable, List, Optional, Dict, Tuple
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch import randperm
from torch._utils import _accumulate
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ttab.api import PyTorchDataset, Batch, GroupBatch
from ttab.loads.datasets.utils.preprocess_toolkit import get_transform
from ttab.loads.datasets.dataset_shifts import (
    NoShiftedData,
    SyntheticShiftedData,
    NaturalShiftedData,
    TemporalShiftedData,
)

dataset_statistics = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "n_classes": 10,
    },
    "cifar100": {
        "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        "std": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        "n_classes": 100,
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "n_classes": 1000,
    },
    "officehome": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "n_classes": 65,
    },
    "pacs": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "n_classes": 7,
    },
    "waterbirds": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "n_classes": 2,
    },
    "coloredmnist": {
        "mean": (0.1307, 0.1307, 0.0),
        "std": (0.3081, 0.3081, 0.3081),
        "n_classes": 2,
    },
}


class WrapperDataset(PyTorchDataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device: str = "cuda"):
        # init other utility functions.
        super().__init__(
            dataset,
            device=device,
            prepare_batch=WrapperDataset.prepare_batch,
            num_classes=None,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class CIFARDataset(PyTorchDataset):
    """A class to load different CIFAR datasets for training and testing.

    CIFAR10-C/CIFAR100-C: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
        https://arxiv.org/abs/1903.12261

    CIFAR10.1: Do CIFAR-10 Classifiers Generalize to CIFAR-10?
        https://arxiv.org/abs/1806.00451
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        split: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        input_size: int = None,
        data_size: int = None,
    ):

        # setup data.
        if "10" in data_name and "100" not in data_name:
            num_classes = dataset_statistics["cifar10"]["n_classes"]
            normalize = {
                "mean": dataset_statistics["cifar10"]["mean"],
                "std": dataset_statistics["cifar10"]["std"],
            }
            dataset_fn = datasets.CIFAR10
        elif "100" in data_name:
            num_classes = dataset_statistics["cifar100"]["n_classes"]
            normalize = {
                "mean": dataset_statistics["cifar100"]["mean"],
                "std": dataset_statistics["cifar100"]["std"],
            }
            dataset_fn = datasets.CIFAR100
        else:
            raise NotImplementedError(f"invalid data_name={data_name}.")

        # data transform.
        if input_size is None:
            input_size = 32
        is_train = True if split == "train" else False
        augment = True if data_augment else False
        if augment:
            scale_size = 40 if input_size == 32 else None
        else:
            scale_size = input_size

        # self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform = get_transform(
            data_name,
            input_size=input_size,
            scale_size=scale_size,
            normalize=normalize,
            augment=augment,
        )
        self.target_transform = None

        # init dataset.
        basic_conf = {
            "root": root,
            "train": is_train,
            "transform": self.transform,
            "target_transform": self.target_transform,
            "download": True,
        }

        if "deterministic" in data_name:
            data_shift_class = functools.partial(
                NoShiftedData, data_name=data_name
            )  # deterministic data is directly loaded from extrinsic files.

        # basic check.
        assert data_shift_class is not None, "data_shift_class is required."

        # configure dataset.
        clean_dataset = dataset_fn(**basic_conf)
        num_nsamples = len(clean_dataset) if data_size is None else data_size
        if issubclass(data_shift_class.func, NoShiftedData):
            if "deterministic" in data_name:
                # get names
                # support string like "cifar10_c_deterministic-gaussian_noise-5"
                _new_data_names = data_name.split("_", 2)
                _shift_name = _new_data_names[-1].split("-")[1]
                _shift_degree = _new_data_names[-1].split("-")[-1]

                # get data
                data_raw = self._load_deterministic_cifar_c(
                    root, _shift_name, _shift_degree
                )

                # construct data_class
                dataset = VisionImageDataset(
                    data=data_raw[:num_nsamples],
                    targets=clean_dataset.targets[:num_nsamples],
                    classes=clean_dataset.classes,
                    class_to_index=clean_dataset.class_to_idx,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            else:
                dataset = VisionImageDataset(
                    data=clean_dataset.data[:num_nsamples],
                    targets=clean_dataset.targets[:num_nsamples],
                    classes=clean_dataset.classes,
                    class_to_index=clean_dataset.class_to_idx,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            dataset = data_shift_class(dataset=dataset)
        elif issubclass(data_shift_class.func, SyntheticShiftedData):
            dataset = VisionImageDataset(
                data=clean_dataset.data[:num_nsamples],
                targets=clean_dataset.targets[:num_nsamples],
                classes=clean_dataset.classes,
                class_to_index=clean_dataset.class_to_idx,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            dataset.apply_corruption()
        elif issubclass(data_shift_class.func, NaturalShiftedData):
            dataset = VisionImageDataset(
                data=clean_dataset.data[:num_nsamples],
                targets=clean_dataset.targets[:num_nsamples],
                classes=clean_dataset.classes,
                class_to_index=clean_dataset.class_to_idx,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            new_indices = list([x for x in range(0, len(dataset.data))])
            dataset.dataset._replace_indices(new_indices)
        elif issubclass(data_shift_class.func, TemporalShiftedData):
            dataset = VisionImageDataset(
                data=clean_dataset.data[:num_nsamples],
                targets=clean_dataset.targets[:num_nsamples],
                classes=clean_dataset.classes,
                class_to_index=clean_dataset.class_to_idx,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            new_indices = list([x for x in range(0, len(dataset.data))])
            dataset.dataset._replace_indices(new_indices)
        else:
            NotImplementedError

        # init other utility functions.
        super().__init__(
            dataset,
            device=device,
            prepare_batch=CIFARDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    def _download_cifar_c(self):
        pass

    def _load_deterministic_cifar_c(self, root, shift_name, shift_degree):
        domain_path = os.path.join(root + "_c", shift_name + ".npy")

        if not os.path.exists(domain_path):
            # TODO: should we enable an automatic configurations / at least a detailed instruction.
            # download data from website: https://zenodo.org/record/2535967#.YxS6D-wzY-R
            pass

        data_raw = np.load(domain_path)
        data_raw = data_raw[(int(shift_degree) - 1) * 10000 : int(shift_degree) * 10000]
        return data_raw

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class ImageNetDataset(PyTorchDataset):
    def __init__(
        self,
        root,
        data_name: str,
        split,
        device="cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        # setup data.
        is_train = True if split == "train" else False
        self.transform = get_transform(
            "imagenet", augment=any([is_train, data_augment]), color_process=False
        )
        self.target_transform = None
        num_classes = dataset_statistics["imagenet"]["n_classes"]

        if "deterministic" in data_name:
            data_shift_class = functools.partial(
                NoShiftedData, data_name=data_name
            )  # deterministic data is directly loaded from extrinsic files.

        # basic check.
        assert data_shift_class is not None, "data_shift_class is required."

        # configure dataset.
        if issubclass(data_shift_class.func, NoShiftedData):
            if "deterministic" in data_name:
                _new_data_names = data_name.split(
                    "_", 2
                )  # support string like "cifar10_c_deterministic-gaussian_noise-5"
                _shift_name = _new_data_names[-1].split("-")[1]
                _shift_degree = _new_data_names[-1].split("-")[-1]

                validdir = os.path.join(root, "imagenet-c", _shift_name, _shift_degree)
                dataset = ImageFolderDataset(
                    root=validdir,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            else:
                validdir = os.path.join(root, "val")
                dataset = ImageFolderDataset(
                    root=validdir,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
            dataset = data_shift_class(dataset=dataset)
        elif issubclass(data_shift_class.func, SyntheticShiftedData):
            validdir = os.path.join(root, "val")
            dataset = ImageFolderDataset(
                root=validdir,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            dataset = data_shift_class(dataset=dataset)
            dataset.apply_corruption()

        super().__init__(
            dataset,
            device=device,
            prepare_batch=ImageNetDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class OfficeHomeDataset(PyTorchDataset):
    """
    A class to load officehome dataset for training and testing.
    Deep Hashing Network for Unsupervised Domain Adaptation: https://paperswithcode.com/paper/deep-hashing-network-for-unsupervised-domain
    """

    DOMAINS: list = ["art", "clipart", "product", "realworld"]

    def __init__(
        self,
        root: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        data_size: int = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_statistics["officehome"]["mean"],
            dataset_statistics["officehome"]["std"],
        )
        num_classes = dataset_statistics["officehome"]["n_classes"]
        self.transform = get_transform(
            "officehome", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data.
        dataset = ImageFolderDataset(
            root=root, transform=self.transform, target_transform=self.target_transform
        )
        if data_size is not None:
            dataset.trim_dataset(data_size)

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=OfficeHomeDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    def split_data(self, fractions: List[float], augment: List[bool], seed: int = 0):
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_statistics["officehome"]["mean"],
            dataset_statistics["officehome"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "officehome",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=OfficeHomeDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class PACSDataset(PyTorchDataset):
    """
    A class to load officehome dataset for training and testing.
    Deep Hashing Network for Unsupervised Domain Adaptation: https://paperswithcode.com/paper/deep-hashing-network-for-unsupervised-domain
    """

    DOMAINS: list = ["art", "cartoon", "photo", "sketch"]

    def __init__(
        self,
        root: str,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
        data_size: int = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_statistics["pacs"]["mean"],
            dataset_statistics["pacs"]["std"],
        )
        num_classes = dataset_statistics["pacs"]["n_classes"]
        self.transform = get_transform(
            "pacs", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data.
        dataset = ImageFolderDataset(
            root=root, transform=self.transform, target_transform=self.target_transform
        )
        if data_size is not None:
            dataset.trim_dataset(
                data_size
            )  # trim indices, so it will also control new data.

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=OfficeHomeDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    def split_data(self, fractions: List[float], augment: List[bool], seed: int = 0):
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_statistics["officehome"]["mean"],
            dataset_statistics["officehome"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "officehome",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=OfficeHomeDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class ColoredMNIST(PyTorchDataset):
    def __init__(
        self,
        root: str,
        data_name: str,
        split: str,
        color_flip_prob: float = None,
        device: str = "cuda",
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        self.split = split
        # set up transform
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        dataset_statistics["coloredmnist"]["mean"],
                        dataset_statistics["coloredmnist"]["std"],
                    ),
                ]
            )
            self.target_transform = None
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        dataset_statistics["coloredmnist"]["mean"],
                        dataset_statistics["coloredmnist"]["std"],
                    ),
                ]
            )
            self.target_transform = None

        # set up data.
        original_dataset = datasets.mnist.MNIST(
            root,
            train=True,
            download=True,
        )
        num_classes = 2

        # init dataset.
        assert issubclass(
            data_shift_class.func, SyntheticShiftedData
        ), "ColoredMNIST belongs to synthetic shift type."
        dataset = data_shift_class(
            dataset=original_dataset,
            color_flip_prob=color_flip_prob
            if color_flip_prob is not None
            else self._default_color_flip_prob,
        )

        dataset = dataset.prepare_colored_mnist(
            transform=self.transform, target_transform=self.target_transform
        )

        # init other utility functions.
        super().__init__(
            dataset=dataset[split],
            device=device,
            prepare_batch=ColoredMNIST.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)

    @property
    def _default_color_flip_prob(self):
        if self.split == "train":
            return 0.1
        elif self.split == "test":
            return 0.9


class WBirdsDataset(PyTorchDataset):
    def __init__(
        self,
        root: str,
        split: str,
        device: str,
        data_augment: bool = False,
        data_shift_class: Optional[Callable] = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(
            dataset_statistics["waterbirds"]["mean"],
            dataset_statistics["waterbirds"]["std"],
        )
        num_classes = dataset_statistics["waterbirds"]["n_classes"]
        self.transform = get_transform(
            "waterbirds", normalize=normalize, augment=data_augment, color_process=False
        )
        self.target_transform = None

        # set up data
        assert os.path.exists(
            root
        ), f"{root} does not exist yet, please generate the dataset first."

        # read in metadata.
        metadata_df = pd.read_csv(os.path.join(root, "metadata.csv"))

        split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }
        split_array = metadata_df["split"].values
        filename_array = metadata_df["img_filename"].values[
            split_array == split_dict[split]
        ]

        # Get the y values
        y_array = metadata_df["y"].values[split_array == split_dict[split]]
        self.target_name = "waterbird_complete95"

        # waterbirds dataset has only one confounder: places.
        confounder_array = metadata_df["place"].values[split_array == split_dict[split]]
        self.n_confounders = 1
        self.confounder_names = ["forest2water2"]
        # map to groups
        self.n_groups = pow(2, 2)
        group_array = (y_array * (self.n_groups / 2) + confounder_array).astype("int")
        self._group_counts = (
            (torch.arange(self.n_groups).unsqueeze(1) == torch.LongTensor(group_array))
            .sum(1)
            .float()
        )

        classes = [
            "0 - landbird",
            "1 - waterbird",
        ]
        class_to_index = {
            "0 - landbird": 0,
            "1 - waterbird": 1,
        }

        dataset = ConfounderDataset(
            root=root,
            data=None,
            filename_array=filename_array,
            targets=list(y_array),
            group_array=group_array,
            classes=classes,
            class_to_index=class_to_index,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=WBirdsDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    def split_dataset(self, fractions: List[float], augment: List[bool], seed: int = 0):
        """This function is used to divide the dataset into two or more than two splits."""
        assert len(fractions) == len(augment)
        lengths = [int(f * len(self.dataset)) for f in fractions]
        lengths[0] += len(self.dataset) - sum(lengths)

        indices = randperm(
            sum(lengths), generator=torch.Generator().manual_seed(seed)
        ).tolist()
        sub_indices = [
            indices[offset - length : offset]
            for offset, length in zip(_accumulate(lengths), lengths)
        ]

        normalize = transforms.Normalize(
            dataset_statistics["waterbirds"]["mean"],
            dataset_statistics["waterbirds"]["std"],
        )
        sub_datasets = [
            SubDataset(
                data=self.dataset.data,
                targets=self.dataset.targets,
                indices=sub_indices[i],
                transform=get_transform(
                    "waterbirds",
                    normalize=normalize,
                    augment=augment[i],
                    color_process=False,
                ),
                target_transform=None,
            )
            for i in range(len(sub_indices))
        ]

        return [
            PyTorchDataset(
                dataset=dataset,
                device=self._device,
                prepare_batch=WBirdsDataset.prepare_batch,
                num_classes=self.num_classes,
            )
            for dataset in sub_datasets
        ]

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.num_classes)
        c = group_idx % (self.n_groups // self.num_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name

    def group_counts(self):
        return self._group_counts

    @staticmethod
    def prepare_batch(batch, device):
        return GroupBatch(*batch).to(device)


class CustomDataset(PyTorchDataset):
    def __init__(
        self,
        config,
        device: str = "cuda",
        data_augment: bool = True,
        data_shift_class: Optional[Callable] = None,
    ):
        # some basic dataset configuration.
        normalize = transforms.Normalize(config.mean, config.std)
        num_classes = config.n_classes

        self.transform = get_transform(
            "custom_dataset",
            normalize=normalize,
            augment=data_augment,
            color_process=False,
        )
        self.target_transform = None

        # set up data.
        dataset = ImageFolderDataset(
            root=config.data_path,
            transform=self.transform,
            target_transform=self.target_transform,
        )

        if data_shift_class is not None:
            dataset = data_shift_class(dataset=dataset)

        super().__init__(
            dataset=dataset,
            device=device,
            prepare_batch=CustomDataset.prepare_batch,
            num_classes=num_classes,
        )

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


class MergeMultiDataset(PyTorchDataset):
    """MergeMultiDataset combines a list of sub-datasets as one augmented dataset"""

    def __init__(self, datasets: List[PyTorchDataset]):
        self.datasets = datasets
        self.device = datasets[0]._device

        # some basic dataset configuration TODO: add a warning to the log
        self.transform = datasets[0].transform
        self.target_transform = datasets[0].target_transform
        num_classes = datasets[0].num_classes

        merged_dict = self.merge_datasets(datasets)
        if any(isinstance(self.datasets[0], t) for t in [WBirdsDataset]):
            dataset = ConfounderDataset(
                root=None,
                data=merged_dict["merged_data"],
                filename_array=None,
                targets=merged_dict["merged_targets"],
                group_array=merged_dict["merged_group_arrays"],
                classes=datasets[0].dataset.classes
                if datasets[0].dataset.classes is not None
                else None,
                class_to_index=datasets[0].dataset.class_to_index
                if datasets[0].dataset.class_to_index is not None
                else None,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        else:
            dataset = VisionImageDataset(
                data=merged_dict["merged_data"],
                targets=merged_dict["merged_targets"],
                classes=datasets[0].dataset.classes
                if datasets[0].dataset.classes is not None
                else None,
                class_to_index=datasets[0].dataset.class_to_index
                if datasets[0].dataset.class_to_index is not None
                else None,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        dataset._replace_indices(merged_dict["merged_indices"])

        super().__init__(
            dataset=dataset,
            device=self.device,
            prepare_batch=self.datasets[0].prepare_batch,
            num_classes=num_classes,
        )

    def merge_datasets(self, datasets: List[PyTorchDataset]):

        merged_data = []
        merged_targets = []
        merged_indices = []
        if isinstance(datasets[0], WBirdsDataset):
            # if hasattr(datasets[0].dataset.dataset, "group_array"):
            merged_group_arrays = []
        else:
            merged_group_arrays = None
        cumulative_size = 0

        all_has_same_type = all(
            isinstance(dataset, type(datasets[0])) for dataset in datasets
        )
        assert (
            all_has_same_type
        ), "All datasets to be merged should be of the same type."

        if isinstance(datasets[0].dataset.data, list):
            for dataset in datasets:
                merged_data += dataset.dataset.data
                merged_targets += dataset.dataset.targets
                if isinstance(datasets[0], WBirdsDataset):
                    # if hasattr(dataset.dataset.dataset, "group_array"):
                    merged_group_arrays.append(dataset.dataset.dataset.group_array)
                merged_indices += [i + cumulative_size for i in dataset.dataset.indices]
                cumulative_size += len(dataset.dataset.data)
            merged_dict = {
                "merged_data": merged_data,
                "merged_targets": merged_targets,
                "merged_indices": merged_indices,
                "merged_group_arrays": None
                if merged_group_arrays is None
                else np.concatenate(merged_group_arrays, axis=0),
            }
            return merged_dict
        elif isinstance(datasets[0].dataset.data, np.ndarray):
            for dataset in datasets:
                merged_data.append(dataset.dataset.data)
                merged_targets += dataset.dataset.targets
                if isinstance(datasets[0], WBirdsDataset):
                    # if hasattr(dataset.dataset.dataset, "group_array"):
                    merged_group_arrays.append(dataset.dataset.dataset.group_array)
                merged_indices += [i + cumulative_size for i in dataset.dataset.indices]
                cumulative_size += len(dataset.dataset.data)
            merged_dict = {
                "merged_data": np.concatenate(merged_data, axis=0),
                "merged_targets": merged_targets,
                "merged_indices": merged_indices,
                "merged_group_arrays": None
                if merged_group_arrays is None
                else np.concatenate(merged_group_arrays, axis=0),
            }
            return merged_dict
        else:
            raise NotImplementedError

    def replace_indices(
        self, indices_pattern: str = "original", new_indices=None, random_seed=None
    ) -> None:
        self.dataset._transform_indices(
            indices_pattern=indices_pattern,
            new_indices=new_indices,
            random_seed=random_seed,
        )

    @staticmethod
    def prepare_batch(batch, device):
        return Batch(*batch).to(device)


"""auxillary (empty) dataset class."""


class ImageFolderDataset(torch.utils.data.Dataset):
    EXTENSIONS = (
        ".jpg",
        ".jpeg",
        ".png",
        ".ppm",
        ".bmp",
        ".pgm",
        ".tif",
        ".tiff",
        ".webp",
    )
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

        Support conventional image formats when reading local images: ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp']
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # prepare info
        self.transform = transform
        self.target_transform = target_transform

        self.loader = datasets.folder.default_loader

        # setup of data and targets
        self.classes, self.class_to_index = self._find_classes(root)
        # for saving cpu memory, we only save the path to images here in self.data
        self.data, self.targets = self._make_dataset(
            root=root,
            class_to_idx=self.class_to_index,
            is_allowed_file=self._has_file_allowed_extension,
        )
        self.data_size = len(self.data)
        self.indices = list([x for x in range(0, self.data_size)])

        self.label_statistics = self._count_label_statistics(labels=self.targets)
        # print label statistics---------------------------------------------------------
        # for (i, v) in self.label_statistics.items():
        #     print(f"category={i}: {v}.\n")

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img = self.data[data_idx]
        if isinstance(img, str):
            img = self._load_allowed_images(img)
        img = Image.fromarray(img)
        target = self.targets[data_idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.indices)
        # return len(self.data)

    def _find_classes(self, root) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            root (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [cls.name for cls in os.scandir(root) if cls.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def _has_file_allowed_extension(self, filename: str) -> bool:
        """Checks if a file is an allowed extension."""
        return filename.lower().endswith(self.EXTENSIONS)

    def _load_allowed_images(self, dir: str) -> List[np.ndarray]:
        img = np.array(self.loader(dir))
        assert img.ndim == 3 and img.shape[2] == 3, "The image shape must be correct."
        return img

    # def _load_npy_data(
    #     self,
    #     dir: str,
    #     ) -> List[np.ndarray]:
    #     """
    #     data in .npy file should have four dimensions: [num_samples, width, height, num_channels]
    #     """
    #     data = np.load(dir)
    #     assert data.ndim == 4 and data.shape[-1] == 3, "The data shape must be correct."

    #     return [data[i] for i in range(data.shape[0])]

    def _get_load_function(self, data_format):
        # return {
        #     "jpg": self._load_allowed_images,
        #     "jpeg": self._load_allowed_images,
        #     "png": self._load_allowed_images,
        #     "bmp": self._load_allowed_images,
        #     "npy": self._load_npy_data,
        # }[data_format]
        return self._load_allowed_images

    def _make_dataset(
        self,
        root: str,
        class_to_idx: Dict[str, int],
        is_allowed_file: Callable[[str], bool],
    ):
        imgs = []
        labels = []
        root = os.path.expanduser(root)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for dir, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(dir, fname)
                    if is_allowed_file(path):
                        imgs.append(path)
                        labels.append(class_index)
                    else:
                        raise NotImplementedError(
                            f"The extension = {self._get_file_extension(fname)} is not supported yet."
                        )
                    # if is_allowed_file(path):
                    #     load_helper = self._get_load_function(
                    #         self._get_file_extension(fname)
                    #     )
                    #     imgs.append(load_helper(path))
                    #     labels.append(class_index)
                    # else:
                    #     raise NotImplementedError(
                    #         f"The extension = {self._get_file_extension(fname)} is not supported yet."
                    #     )

        return imgs, labels

    def _transform_indices(
        self, indices_pattern: str, new_indices=None, random_seed=None
    ):
        if indices_pattern == "original":
            pass
        elif indices_pattern == "random_shuffle":
            rng = np.random.default_rng(random_seed)
            rng.shuffle(self.indices)
        elif indices_pattern == "new":
            assert new_indices is not None, "new_indices is required to be not None."
            self._replace_indices(new_indices=new_indices)
        else:
            raise NotImplementedError

    def _replace_indices(self, new_indices: List[int]):
        self.indices = new_indices
        self.data_size = len(self.indices)

    def _get_file_extension(self, fname):
        return fname.split(".")[-1]

    def _count_label_statistics(self, labels: list) -> dict:
        """
        This function returns the statistics of each image category.
        """
        label_statistics = {}

        if self.class_to_index is not None:
            for k, v in sorted(self.class_to_index.items(), key=lambda item: item[1]):
                num_occurrence = labels.count(v)
                label_statistics[k] = num_occurrence
        else:
            # get the number of categories.
            num_categories = len(set(labels))
            for i in range(num_categories):
                num_occurrence = labels.count(i)
                label_statistics[str(i)] = num_occurrence

        return label_statistics

    def trim_dataset(self, data_size):
        """trim dataset given a data size"""
        assert data_size <= len(
            self
        ), "given data size should be smaller than the original data size."
        self.indices = self.indices[:data_size]
        self.data_size = len(self.indices)


class VisionImageDataset(ImageFolderDataset):
    """This is a modified version of TorchImageDataset.
    VisionImageDataset supports dataset downloaded from torchvision library, and all other datasets that do not need to load from the disk.
    """

    def __init__(
        self,
        data: np.ndarray,
        targets: list,
        classes: Optional[list] = None,
        class_to_index: Optional[dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.data = data
        self.targets = targets
        self.data_size = self._get_data_size
        self.indices = list([x for x in range(0, self.data_size)])
        if classes is not None:
            assert class_to_index is not None, "class_to_index needs to be specified "
            self.classes = classes
            self.class_to_index = class_to_index

        self.transform = transform
        self.target_transform = target_transform
        self.loader = datasets.folder.default_loader

        self.label_statistics = self._count_label_statistics(labels=self.targets)
        # print label statistics---------------------------------------------------------
        # for (i, v) in self.label_statistics.items():
        #     print(f"category={i}: {v}.\n")

    @property
    def _get_data_size(self):
        data_size = (
            len(self.data) if isinstance(self.data, list) else self.data.shape[0]
        )
        return data_size


class MetadataImageDataset(ImageFolderDataset):
    """Deprecated! This is a modified version of ImageFolderDataset.
    MetaDataImageDataset supports using metadata files to load dataset.
    """

    def __init__(
        self,
        root,
        filename_array,
        targets,
        classes: Optional[list] = None,
        class_to_index: Optional[dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.loader = datasets.folder.default_loader
        self.data = self._make_dataset(
            root=root,
            filename_array=filename_array,
            is_allowed_file=self._has_file_allowed_extension,
        )
        self.targets = targets
        self.data_size = self._get_data_size
        self.indices = list([x for x in range(0, self.data_size)])
        self.classes = classes
        self.class_to_index = class_to_index

        self.transform = transform
        self.target_transform = target_transform

        self.label_statistics = self._count_label_statistics(labels=self.targets)

    def _make_dataset(
        self,
        root,
        filename_array,
        is_allowed_file: Callable[[str], bool],
    ):
        imgs = []
        root = os.path.expanduser(root)

        for i in range(len(filename_array)):
            img_filename = filename_array[i]
            if is_allowed_file(img_filename):
                abs_imgpath = os.path.join(root, img_filename)
                load_helper = self._get_load_function(
                    self._get_file_extension(img_filename)
                )
                imgs.append(load_helper(abs_imgpath))
            else:
                raise NotImplementedError(
                    f"The extension = {self._get_file_extension(img_filename)} is not supported yet."
                )

        return imgs

    @property
    def _get_data_size(self):
        return len(self.data)


class ConfounderDataset(ImageFolderDataset):
    """This class is designed for datasets that consider confounders such as waterbirds."""

    def __init__(
        self,
        root,
        data,
        filename_array,
        targets,
        group_array,
        classes: Optional[list] = None,
        class_to_index: Optional[dict] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.loader = datasets.folder.default_loader
        assert (data is not None) or (
            filename_array is not None
        ), "Pls load data source."
        if data is None:
            self.data = self._make_dataset(
                root=root,
                filename_array=filename_array,
                is_allowed_file=self._has_file_allowed_extension,
            )
        else:
            self.data = data
        self.targets = targets
        self.group_array = group_array
        self.data_size = self._get_data_size
        self.indices = list([x for x in range(0, self.data_size)])
        self.classes = classes
        self.n_classes = len(self.classes)
        self.class_to_index = class_to_index

        self.transform = transform
        self.target_transform = target_transform

        self.label_statistics = self._count_label_statistics(labels=self.targets)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        img_path = self.data[data_idx]
        img = self._load_allowed_images(img_path)
        img = Image.fromarray(img)
        target = self.targets[data_idx]
        group = self.group_array[data_idx]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, group

    def _make_dataset(
        self,
        root,
        filename_array,
        is_allowed_file: Callable[[str], bool],
    ):
        imgs = []
        root = os.path.expanduser(root)

        for i in range(len(filename_array)):
            img_filename = filename_array[i]
            if is_allowed_file(img_filename):
                abs_imgpath = os.path.join(root, img_filename)
                imgs.append(abs_imgpath)
            else:
                raise NotImplementedError(
                    f"The extension = {self._get_file_extension(img_filename)} is not supported yet."
                )
            # if is_allowed_file(img_filename):
            #     abs_imgpath = os.path.join(root, img_filename)
            #     load_helper = self._get_load_function(
            #         self._get_file_extension(img_filename)
            #     )
            #     imgs.append(load_helper(abs_imgpath))
            # else:
            #     raise NotImplementedError(
            #         f"The extension = {self._get_file_extension(img_filename)} is not supported yet."
            #     )

        return imgs

    @property
    def _get_data_size(self):
        return len(self.data)


# TODO: improve getitem function to support flexible loading (img_path or npy).
class SubDataset(torch.utils.data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, data, targets, indices, transform, target_transform) -> None:
        self.data = data
        self.targets = targets
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform
        self.loader = datasets.folder.default_loader

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        img_path = self.data[data_idx]
        img = self._load_allowed_images(img_path)
        # img = self.data[self.indices[idx]]
        img = Image.fromarray(img)
        target = self.targets[self.indices[idx]]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.indices)

    def _load_allowed_images(self, dir: str) -> List[np.ndarray]:
        img = np.array(self.loader(dir))
        assert img.ndim == 3 and img.shape[2] == 3, "The image shape must be correct."
        return img
