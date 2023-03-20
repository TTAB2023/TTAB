# -*- coding: utf-8 -*-

# 1. This file collects significant hyperparameters that control the behaviour of different model architectures.
# 2. We are only concerned about model-related hyperparameters here, e.g., optimizer and scheduler.
# 3. We provide default hyperparameters if users have no idea how to set up reasonable values.

model_defaults = {
    "resnet18": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "resnet20": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "resnet26": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "wideresnet40_2": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "wideresnet40_4": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "wideresnet28_10": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
    "resnet50": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": (224, 224),
    },
    "vit_base_patch16_224": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": (224, 224),
    },
    "vit_tiny_patch16_224": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": (224, 224),
    },
    "vit_small_patch16_224": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": (224, 224),
    },
    "Hendrycks2020AugMix_WRN": {
        "model_kwargs": {"pretrained": True},
        "target_resolution": None,
    },
}
