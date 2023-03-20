# -*- coding: utf-8 -*-
import copy
import os
import functools
from typing import List

import numpy as np
import torch

from ttab.api import PyTorchDataset
from ttab.loads.datasets.datasets import (
    CIFARDataset,
    ImageNetDataset,
    OfficeHomeDataset,
    ColoredMNIST,
    WBirdsDataset,
    MergeMultiDataset,
    ImageFolderDataset,
    PACSDataset,
)
from ttab.loads.datasets.cifar import CIFAR10_1, CIFARSyntheticShift, LabelShiftedCIFAR
from ttab.loads.datasets.imagenet import ImageNetValNaturalShift, ImageNetSyntheticShift
from ttab.loads.datasets.mnist import ColoredSyntheticShift
from ttab.loads.datasets.dataset_shifts import (
    NoShiftedData,
    SyntheticShiftedData,
    NaturalShiftedData,
    TemporalShiftedData,
    NaturalShiftProperty,
    NoShiftProperty,
)
from ttab.loads.datasets.dataset_sampling import DatasetSampling
import ttab.loads.datasets.loaders as loaders

from ttab.scenarios import (
    HomogeneousNoMixture,
    HeterogeneousNoMixture,
    InOutMixture,
    CrossMixture,
    TestDomain,
)


class MergeMultiTestDatasets(object):
    def _intra_shuffle_dataset(
        self, dataset: PyTorchDataset, random_seed=None
    ) -> PyTorchDataset:
        """shuffle the dataset."""
        dataset.replace_indices(
            indices_pattern="random_shuffle", random_seed=random_seed
        )
        return dataset

    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/f44cf4281944fae46cdce1b8bc7cde3e7c44bd70/experiment.py
    def _intra_non_iid_shift(
        self,
        dataset: PyTorchDataset,
        non_iid_pattern: str = "class_wise_over_domain",
        non_iid_ness: float = 0.1,
        random_seed=None,
    ) -> PyTorchDataset:
        """make iid dataset non-iid through applying dirichlet distribution."""
        rng = np.random.default_rng(random_seed)
        new_indices = []
        targets = dataset.dataset.targets

        if non_iid_pattern == "class_wise_over_domain":
            dirichlet_numchunks = dataset.num_classes
            min_size = -1
            N = len(dataset)
            min_size_threshold = (
                10  # hyperparameter. if conf.args.dataset in ['tinyimagenet'] else 10
            )
            while (
                min_size < min_size_threshold
            ):  # prevent any chunk having too less data
                idx_batch = [[] for _ in range(dirichlet_numchunks)]
                idx_batch_cls = [
                    [] for _ in range(dirichlet_numchunks)
                ]  # contains data per each class
                for k in range(dataset.num_classes):
                    targets_np = torch.Tensor(targets).numpy()
                    idx_k = np.where(targets_np == k)[0]
                    # rng.shuffle(idx_k)
                    # proportions = rng.dirichlet(
                    #     np.repeat(non_iid_ness, dirichlet_numchunks)
                    # )
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(
                        np.repeat(non_iid_ness, dirichlet_numchunks)
                    )

                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < N / dirichlet_numchunks)
                            for p, idx_j in zip(proportions, idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                    ]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

                    # store class-wise data
                    for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                        idx_j.append(idx)

            sequence_stats = []
            # create temporally correlated toy dataset by shuffling classes
            for chunk in idx_batch_cls:
                cls_seq = list(range(dataset.num_classes))
                # rng.shuffle(cls_seq)
                np.random.shuffle(cls_seq)
                for cls in cls_seq:
                    idx = chunk[cls]
                    # new_data.extend([data[i] for i in idx])
                    # new_targets.extend([targets[i] for i in idx])
                    new_indices.extend(idx)
                    sequence_stats.extend(list(np.repeat(cls, len(idx))))

            # trim data if num_sample is smaller than the original data size
            # TODO: currently, we simply set num_samples as the data size.
            num_samples = len(new_indices)
            new_indices = new_indices[:num_samples]
            # new_data = np.concatenate(new_data[:num_samples], axis=0)
            # new_targets = new_targets[:num_samples]

            # replace dataset with new data.
            # dataset.dataset.data = new_data
            # dataset.dataset.targets = new_targets
            dataset.replace_indices(indices_pattern="new", new_indices=new_indices)
        return dataset

    def _merge_datasets(self, datasets: List[PyTorchDataset]) -> PyTorchDataset:
        return MergeMultiDataset(datasets=datasets)

    def _merge_two_datasets(
        self, left_dataset: PyTorchDataset, right_dataset: PyTorchDataset, ratio
    ) -> PyTorchDataset:
        return self._merge_datasets([left_dataset, right_dataset])

    def merge(
        self,
        test_case,
        test_domains,
        test_datasets: List[PyTorchDataset],
        in_test_dataset: PyTorchDataset = None,
        random_seed=None,
    ) -> PyTorchDataset:
        if isinstance(test_case.inter_domain, HomogeneousNoMixture):
            return self._merge_datasets(
                [
                    self._intra_shuffle_dataset(test_dataset, random_seed)
                    if test_case.intra_domain_shuffle
                    else test_dataset
                    for test_dataset in test_datasets
                ]
            )
        elif isinstance(test_case.inter_domain, HeterogeneousNoMixture):
            return self._merge_datasets(
                [
                    self._intra_non_iid_shift(
                        dataset=dataset,
                        non_iid_pattern=test_case.inter_domain.non_iid_pattern,
                        non_iid_ness=test_case.inter_domain.non_iid_ness,
                        random_seed=random_seed,
                    )
                    for dataset in test_datasets
                ]
            )
        elif isinstance(test_case.inter_domain, InOutMixture):
            # TODO: need to consider the `ratio`` in the class (and with seed control).
            return self._merge_datasets(
                [
                    self._intra_shuffle_dataset(
                        self._merge_two_datasets(
                            in_test_dataset,
                            test_dataset,
                            ratio=test_case.inter_domain.ratio,
                        ),
                        random_seed=random_seed,
                    )
                    for test_dataset in test_datasets[1:]
                ]
            )
        elif isinstance(test_case.inter_domain, CrossMixture):
            raise NotImplementedError


class ConstructTestDataset(object):
    # def __init__(self, data_path, base_data_name, seed, device) -> None:
    #     self.data_path = data_path
    #     self.base_data_name = base_data_name
    #     self.seed = seed
    #     self.device = device
    def __init__(self, config) -> None:
        self.meta_conf = config
        self.data_path = self.meta_conf.data_path
        self.base_data_name = self.meta_conf.base_data_name
        self.seed = self.meta_conf.seed
        self.device = self.meta_conf.device

    def get_test_datasets(self, test_domains, data_augment: bool = False):
        """This function defines the target domain dataset(s) for evaluation."""
        helper_fn = self._get_target_domain_helper()
        return [
            DatasetSampling(test_domain).sample(helper_fn(test_domain, data_augment))
            for test_domain in test_domains
        ]

    def get_in_test_dataset(self, test_domain, data_augment: bool = False):
        # get test_domain
        _test_domain = copy.deepcopy(test_domain)
        _test_domain._replace(shift_type="no_shift")
        helper_fn = self._get_target_domain_helper()
        return DatasetSampling(_test_domain).sample(
            helper_fn(_test_domain, data_augment)
        )

    def _get_target_domain_helper(self):
        if "cifar" in self.base_data_name:
            # CIFAR datasets support [no_shift, natural, synthetic].
            helper_fn = self._get_cifar_test_domain_datasets_helper
        elif "imagenet" in self.base_data_name:
            # ImageNet datasets support [no_shift, natural, synthetic].
            helper_fn = self._get_imagenet_test_domain_datasets_helper
        elif "officehome" in self.base_data_name:
            # OfficeHome dataset only supports [natural].
            helper_fn = self._get_officehome_test_domain_datasets_helper
        elif "pacs" in self.base_data_name:
            # OfficeHome dataset only supports [natural].
            helper_fn = self._get_pacs_test_domain_datasets_helper
        elif "mnist" in self.base_data_name:
            # This benchmark only supports ColoredMNIST now, which is synthetic shift.
            helper_fn = self._get_mnist_test_domain_datasets_helper
        elif "waterbirds" in self.base_data_name:
            helper_fn = self._get_waterbirds_test_domain_datasets_helper
        return helper_fn

    def _get_cifar_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        # get data_shift_class
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "natural":
            if "shiftedlabel" in test_domain.data_name:
                data_shift_class = functools.partial(
                    NaturalShiftedData,
                    data_name=test_domain.data_name,
                    new_data=LabelShiftedCIFAR(
                        root=os.path.join(self.data_path, test_domain.base_data_name),
                        data_name=test_domain.data_name,
                        shift_type=test_domain.shift_property.version,
                        train=False,
                        param=self.meta_conf.label_shift_param,
                        data_size=self.meta_conf.data_size,
                        random_seed=self.seed,
                    ),
                )
            elif test_domain.data_name == "cifar10_1":
                data_shift_class = functools.partial(
                    NaturalShiftedData,
                    data_name=test_domain.data_name,
                    new_data=CIFAR10_1(
                        root=os.path.join(self.data_path, test_domain.data_name),
                        data_name=test_domain.data_name,
                        version=test_domain.shift_property.version,
                    ),
                )
            else:
                raise NotImplementedError(f"invalid data_name={test_domain.data_name}.")
        elif test_domain.shift_type == "synthetic":
            assert 1 <= test_domain.shift_property.shift_degree <= 5

            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=functools.partial(
                    CIFARSyntheticShift,
                    corruption_type=test_domain.shift_property.shift_name,
                ),
                version=test_domain.shift_property.version,
                severity=test_domain.shift_property.shift_degree,
            )
        elif (
            test_domain.shift_type == "temporal"
        ):  # TODO: add real-world temporal distribution shift. The relevant temporal distribution shift in the paper is covariate shift + label shift.
            raise NotImplementedError
        else:
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for cifar datasets"
            )

        # create dataset.
        dataset = CIFARDataset(
            root=os.path.join(self.data_path, self.base_data_name),
            data_name=test_domain.data_name,
            split="test",
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            input_size=int(self.meta_conf.model_name.split("_")[-1])
            if "vit" in self.meta_conf.model_name
            else 32,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_imagenet_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "synthetic":
            assert 1 <= test_domain.shift_property.shift_degree <= 5

            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=functools.partial(
                    ImageNetSyntheticShift,
                    corruption_type=test_domain.shift_property.shift_name,
                ),
                version=test_domain.shift_property.version,
                severity=test_domain.shift_property.shift_degree,
            )
        elif test_domain.shift_type == "natural":
            assert test_domain.data_name in [
                "imagenet_a",
                "imagenet_r",
                "imagenet_v2_matched-frequency",
                "imagenet_v2_threshold0.7",
                "imagenet_v2_topimages",
            ]
            data_shift_class = functools.partial(
                NaturalShiftedData,
                data_name=test_domain.data_name,
                new_data=ImageNetValNaturalShift(
                    root=os.path.join(self.data_path, test_domain.base_data_name),
                    data_name=test_domain.data_name,
                    version=test_domain.shift_property.version,
                ),
            )
        else:
            raise NotImplementedError(
                f"invalid shift type={test_domain.shift_type} for ImageNet dataset."
            )

        dataset = ImageNetDataset(
            root=os.path.join(self.data_path, "ILSVRC"),
            data_name=test_domain.data_name,
            split="test",
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )
        return dataset

    def _get_officehome_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        assert (
            test_domain.shift_type == "natural"
        ), "officehome dataset only works when shift_type == natural."

        _data_names = test_domain.data_name.split("_")  # e.g., "officehome_art"
        domain_path = os.path.join(self.data_path, self.base_data_name, _data_names[1])
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=ImageFolderDataset(root=domain_path),
        )

        # new target domain dataset will replace the domain dataset. TODO: add some comments.
        dataset = OfficeHomeDataset(
            root=domain_path,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_pacs_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        assert (
            test_domain.shift_type == "natural"
        ), "officehome dataset only works when shift_type == natural."

        _data_names = test_domain.data_name.split("_")  # e.g., "pacs_art"
        domain_path = os.path.join(self.data_path, self.base_data_name, _data_names[1])
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=ImageFolderDataset(root=domain_path),
        )

        # new target domain dataset will replace the domain dataset. TODO: add some comments.
        dataset = PACSDataset(
            root=domain_path,
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
            data_size=self.meta_conf.data_size,
        )
        return dataset

    def _get_mnist_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        # get data_shift_class
        if test_domain.shift_type == "no_shift":
            data_shift_class = functools.partial(
                NoShiftedData, data_name=test_domain.base_data_name
            )
        elif test_domain.shift_type == "natural":
            raise NotImplementedError
        elif test_domain.shift_type == "synthetic":
            data_shift_class = functools.partial(
                SyntheticShiftedData,
                data_name=test_domain.data_name,
                seed=self.seed,
                synthetic_class=functools.partial(ColoredSyntheticShift),
                version=test_domain.shift_property.version,
            )
        dataset = ColoredMNIST(
            root=os.path.join(self.data_path, "mnist"),
            data_name=test_domain.data_name,
            split="test",
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )

        return dataset

    def _get_waterbirds_test_domain_datasets_helper(
        self, test_domain, data_augment: bool = False
    ):
        assert (
            test_domain.shift_type == "natural"
        ), "waterbirds dataset only works when shift_type == natural."

        domain_path = os.path.join(self.data_path, self.base_data_name)
        data_shift_class = functools.partial(
            NaturalShiftedData,
            data_name=test_domain.data_name,
            new_data=WBirdsDataset(
                root=domain_path,
                split="test",
                device=self.device,
            ).dataset,
        )

        # new target domain dataset will replace the domain dataset. TODO: add some comments.
        dataset = WBirdsDataset(
            root=domain_path,
            split="test",
            device=self.device,
            data_augment=data_augment,
            data_shift_class=data_shift_class,
        )
        return dataset

    def construct_test_dataset(self, scenario, data_augment: bool = False):
        return MergeMultiTestDatasets().merge(
            test_case=scenario.test_case,
            test_domains=scenario.test_domains,
            test_datasets=self.get_test_datasets(scenario.test_domains, data_augment),
            in_test_dataset=self.get_in_test_dataset(
                scenario.test_domains[0], data_augment
            ),
            random_seed=self.seed,
        )

    def construct_test_loader(self, scenario):
        test_dataset = self.construct_test_dataset(scenario, data_augment=False)
        test_dataloader: loaders.BaseLoader = loaders.get_test_loader(
            dataset=test_dataset, scenario=scenario, device=self.device
        )
        return test_dataloader


class ConstructAuxiliaryDataset(ConstructTestDataset):
    def __init__(self, config) -> None:
        super(ConstructAuxiliaryDataset, self).__init__(config)

    def construct_auxiliary_loader(self, scenario, data_augment: bool = True):
        auxiliary_dataset = self.construct_test_dataset(scenario, data_augment)
        auxiliary_dataloader: loaders.BaseLoader = loaders.get_auxiliary_loader(
            dataset=auxiliary_dataset, device=self.device
        )
        return auxiliary_dataloader

    def construct_in_dataset(self, scenario, data_size, data_augment: bool = True):
        """Load source environment as oen of auxiliary datasets"""
        in_data_test_domain = self.get_in_data_test_domain(scenario)
        helper_fn = self._get_target_domain_helper()
        in_dataset = DatasetSampling(in_data_test_domain).sample(
            helper_fn(in_data_test_domain, data_augment)
        )
        # trim dataset. TODO: to improve
        in_dataset.dataset.dataset.trim_dataset(data_size)
        in_dataloader = loaders.BaseLoader = loaders.get_auxiliary_loader(
            dataset=in_dataset, device=self.device
        )
        return in_dataset, in_dataloader

    def get_in_data_test_domain(self, scenario):
        # simply select the first test domain and modify it into new domain for first domain
        test_domain = copy.deepcopy(scenario.test_domains[0])
        if "cifar" in test_domain.base_data_name:
            data_name = test_domain.base_data_name
            return TestDomain(
                base_data_name=test_domain.base_data_name,
                data_name=data_name,
                shift_type="no_shift",
                shift_property=NoShiftProperty(has_shift=False),
            )
        elif test_domain.base_data_name in ["officehome", "pacs"]:
            data_name = (scenario.in_data_name,)
            return TestDomain(
                base_data_name=test_domain.base_data_name,
                data_name=data_name,
                shift_type=test_domain.shift_type,
                shift_property=test_domain.shift_property,
            )
        else:
            raise NotImplementedError
