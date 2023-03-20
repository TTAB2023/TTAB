import math

import torch

from torch import nn

from ttab.loads.models.network import Featurizer, Classifier


class SeqModel(nn.Module):
    def __init__(self, featurizer, classifier) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.network = nn.Sequential(self.featurizer, self.classifier)

    def forward(self, x):
        x = self.network(x)
        return x


def build_model(
    input_shape,
    num_classes,
    nonlinear_classifier=False,
    backbone="resnet50-BN",
    resnet_dropout=0.0,
):
    hparams = {
        "backbone": backbone,
        "resnet_dropout": resnet_dropout,
    }
    featurizer = Featurizer(input_shape, hparams)
    classifier = Classifier(
        featurizer.n_outputs,
        num_classes,
        nonlinear_classifier,
    )
    return SeqModel(featurizer, classifier)
