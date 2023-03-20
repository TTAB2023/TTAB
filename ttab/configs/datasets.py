# -*- coding: utf-8 -*-

# 1. This file collects hyperparameters significant for pretraining and test-time adaptation.
# 2. We are only concerned about dataset-related hyperparameters here, e.g., lr, dataset statistics, and type of corruptions.
# 3. We provide default hyperparameters if users have no idea how to set up reasonable values.

dataset_defaults = {
    "cifar10": {
        "lr": 0.001,
        "statistics": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "n_classes": 10,
        },
        "version": "deterministic",
    },
    "cifar100": {
        "lr": 0.001,
        "statistics": {
            "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            "std": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            "n_classes": 100,
        },
        "version": "deterministic",
    },
    "cifar10_1": {
        "lr": 0.001,
        "statistics": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
            "n_classes": 10,
        },
    },
    "officehome": {
        "lr": 5e-6,
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 65,
        },
    },
    "pacs": {
        "lr": 1e-3,
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 7,
        },
    },
    "coloredmnist": {
        "lr": 0.03,
        "statistics": {
            "n_classes": 2,
        },
    },
    "waterbirds": {
        "lr": 0.001,
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 2,
        },
        "group_counts": [3498, 184, 56, 1057],  # used to compute group ratio.
    },
    "imagenet": {
        "statistics": {
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
            "n_classes": 1000,
        }
    },
}
