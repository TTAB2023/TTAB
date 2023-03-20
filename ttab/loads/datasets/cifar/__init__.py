# -*- coding: utf-8 -*-
import os
import numpy as np

import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url

from ttab.loads.datasets.cifar.synthetic import (
    gaussian_noise,
    shot_noise,
    impulse_noise,
    defocus_blur,
    glass_blur,
    motion_blur,
    zoom_blur,
    snow,
    frost,
    fog,
    brightness,
    contrast,
    elastic_transform,
    pixelate,
    jpeg_compression,
)


"""The dataset loader for CIFAR10.1 (validation set)."""


class CIFAR10_1(object):
    """Borrowed from https://github.com/modestyachts/CIFAR-10.1"""

    stats = {
        "v4": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy",
        },
        "v6": {
            "data": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy",
            "labels": "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy",
        },
    }

    def __init__(self, root, data_name, version):
        version = "v4" if version is None else version
        assert version in ["v4", "v6"]

        self.data_name = data_name
        self.path_data = os.path.join(root, f"cifar10.1_{version}_data.npy")
        self.path_labels = os.path.join(root, f"cifar10.1_{version}_labels.npy")
        self._download(root, version)

        self.data = np.load(self.path_data)
        self.targets = np.load(self.path_labels).tolist()
        self.data_size = len(self.data)

    def _download(self, root, version):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(url=self.stats[version]["data"], root=root)
        download_url(url=self.stats[version]["labels"], root=root)

    def _check_integrity(self) -> bool:
        if os.path.exists(self.path_data) and os.path.exists(self.path_labels):
            return True
        else:
            return False

    def __getitem__(self, index):
        img_array = self.data[index]
        target = self.targets[index]
        return img_array, target

    def __len__(self):
        return self.data_size


# def np_to_png(a, fmt="png", scale=1):
#     a = np.uint8(a)
#     f = io.BytesIO()
#     tmp_img = PILImage.fromarray(a)
#     tmp_img = tmp_img.resize((scale * 32, scale * 32), PILImage.NEAREST)
#     tmp_img.save(f, fmt)
#     return f.getvalue()


"""Some corruptions are referred to https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py"""


class CIFARSyntheticShift(object):
    """The class of synthetic corruptions/shifts introduced in CIFAR_C."""

    def __init__(self, data_name, seed, severity=5, corruption_type=None):
        assert "cifar" in data_name

        self.data_name = data_name
        self.base_data_name = data_name.split("_")[0]
        self.seed = seed
        self.severity = severity
        self.corruption_type = corruption_type
        self.dict_corruption = {
            "gaussian_noise": gaussian_noise,
            "shot_noise": shot_noise,
            "impulse_noise": impulse_noise,
            "defocus_blur": defocus_blur,
            "glass_blur": glass_blur,
            "motion_blur": motion_blur,
            "zoom_blur": zoom_blur,
            "snow": snow,
            "frost": frost,
            "fog": fog,
            "brightness": brightness,
            "contrast": contrast,
            "elastic_transform": elastic_transform,
            "pixelate": pixelate,
            "jpeg_compression": jpeg_compression,
        }
        if corruption_type is not None:
            assert (
                corruption_type in self.dict_corruption.keys()
            ), f"{corruption_type} is out of range"
        self.random_state = np.random.RandomState(self.seed)

    def _apply_corruption(self, pil_img):
        if self.corruption_type is None or self.corruption_type == "all":
            corruption = self.random_state.choice(self.dict_corruption.values())
        else:
            corruption = self.dict_corruption[self.corruption_type]

        return np.uint8(
            corruption(pil_img, random_state=self.random_state, severity=self.severity)
        )

    def apply(self, pil_img):
        return self._apply_corruption(pil_img)


"""Label shift implementations are referred to https://github.com/Angie-Liu/labelshift/blob/master/cifar10_for_labelshift.py"""


class LabelShiftedCIFAR(object):
    """CIFAR-10 dataset with label shift.
    Type of shifts:
        1. knock one shift.
        2. tweak one shift.
        3. dirichlet shift.
        4. minority class shift.
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        shift_type: str,
        train: bool = False,
        param: float = None,
        data_size: int = None,
        target_label: int = None,
        random_seed: int = None,
    ) -> None:
        self.data_name = data_name
        self.shift_type = shift_type
        self.target_label = target_label
        self.data_size = data_size
        self.seed = random_seed

        # init dataset.
        if "10" in data_name and "100" not in data_name:
            dataset_fn = datasets.CIFAR10
            self.num_classes = 10
        elif "100" in data_name:
            dataset_fn = datasets.CIFAR100
            self.num_classes = 100
        else:
            raise NotImplementedError(f"invalid data_name={data_name}.")

        basic_conf = {
            "root": root,
            "train": train,
            "transform": None,
            "target_transform": None,
            "download": True,
        }
        dataset = dataset_fn(**basic_conf)
        raw_targets = dataset.targets

        _data_names = self.shift_type.split("_")
        if len(_data_names) == 1:
            raw_data = dataset.data
        else:
            _shift_name = "_".join(_data_names[1:-1])
            _shift_degree = _data_names[-1]
            raw_data = self._load_deterministic_cifar_c(
                root, _shift_name, _shift_degree
            )

        # create label shift.
        # np.random.seed(random_seed)
        # indices = np.random.permutation(len(raw_targets))
        rng = np.random.default_rng(self.seed)
        if "uniform" in self.shift_type:
            self.data, self.targets = self._apply_uniform_subset_shift(
                data=raw_data,
                targets=raw_targets,
                data_size=self.data_size
                if self.data_size is not None
                else len(self.data),
                random_generator=rng,
            )
        else:
            label_shifter = self._get_label_shifter()
            self.data, self.targets = label_shifter(
                data=raw_data,
                targets=raw_targets,
                param=param,
                random_generator=rng,
            )

    def _get_label_shifter(self):
        if "constant-size-dirichlet" in self.shift_type:
            return self._apply_constant_size_dirichlet_shift
        elif "dirichlet" in self.shift_type:
            return self._apply_dirichlet_shift
        else:
            raise NotImplementedError(f"invalid shift_type={self.shift_type}.")

    def _apply_dirichlet_shift(self, data, targets, param, random_generator=None):
        if param is None:
            param = 4

        alpha = np.ones(self.num_classes) * param
        # prob = np.random.dirichlet(alpha)
        prob = random_generator.dirichlet(alpha)
        params = prob
        # use the maximum prob to decide the total number of training samples
        target_label = np.argmax(params)

        indices_target = [i for i, x in enumerate(targets) if x == target_label]
        num_targets = len(indices_target)
        prob_max = np.amax(params)
        num_data = int(num_targets / prob_max)
        indices_data = []

        for i in range(self.num_classes):
            num_i = int(num_data * params[i])
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [all_indices_i[shuffle_i[i]] for i in range(num_i)]
            indices_data += selected_indices_i

        # shuffle = random_generator.permutation(len(indices_data))
        # shuffled_indices_data = [indices_data[shuffle[i]] for i in range(len(indices_data))]
        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _apply_uniform_subset_shift(
        self, data, targets, data_size, random_generator=None
    ):

        # uniform on all labels
        num_per_class = int(data_size / self.num_classes)
        indices_data = []

        for i in range(self.num_classes):
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [
                all_indices_i[shuffle_i[j]] for j in range(num_per_class)
            ]
            indices_data += selected_indices_i

        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _apply_constant_size_dirichlet_shift(
        self, data, targets, param, random_generator=None
    ):
        if param is None:
            param = 4

        alpha = np.ones(self.num_classes) * param
        # prob = np.random.dirichlet(alpha)
        prob = random_generator.dirichlet(alpha)
        params = prob
        # use the maximum prob to decide the total number of training samples
        target_label = np.argmax(params)

        indices_target = [i for i, x in enumerate(targets) if x == target_label]
        num_targets = len(
            indices_target
        )  # constant size. Maximum number of samples for a single category.
        prob_sum = np.sum(params)
        indices_data = []

        for i in range(self.num_classes):
            num_i = int(num_targets * params[i] / prob_sum)
            all_indices_i = [t for t, x in enumerate(targets) if x == i]
            shuffle_i = random_generator.permutation(len(all_indices_i))
            selected_indices_i = [all_indices_i[shuffle_i[j]] for j in range(num_i)]
            indices_data += selected_indices_i

        num_samples_to_add = num_targets - len(indices_data)
        if num_samples_to_add > 0:
            left_indices = [
                indice
                for indice in list(range(len(targets)))
                if indice not in indices_data
            ]
            samples = random_generator.choice(
                left_indices, num_samples_to_add, replace=False
            ).tolist()
            indices_data += samples

        # shuffle = np.random.permutation(len(indices_data))
        # shuffle = random_generator.permutation(len(indices_data))
        # shuffled_indices_data = [indices_data[shuffle[i]] for i in range(len(indices_data))]
        shifted_data = data[(indices_data,)]
        shifted_targets = [targets[ele] for ele in indices_data]
        return shifted_data, shifted_targets

    def _load_deterministic_cifar_c(self, root, shift_name, shift_degree):
        domain_path = os.path.join(root + "_c", shift_name + ".npy")

        if not os.path.exists(domain_path):
            # TODO: should we enable an automatic configurations / at least a detailed instruction.
            # download data from website: https://zenodo.org/record/2535967#.YxS6D-wzY-R
            pass

        data_raw = np.load(domain_path)
        data_raw = data_raw[(int(shift_degree) - 1) * 10000 : int(shift_degree) * 10000]
        return data_raw
