# -*- coding: utf-8 -*-
import copy
from typing import List, Optional, Any, Type

import torch

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.base_selection import BaseSelection
from ttab.api import Batch, Dataset, Quality
import ttab.loads.datasets.loaders as loaders
from ttab.utils.logging import Logger
from ttab.utils.timer import Timer


class BaseAdaptation(object):
    def __init__(self, meta_conf, base_model):
        self._meta_conf = copy.deepcopy(meta_conf)
        self._base_model = base_model

        self._prior_safety_check()
        self.initialize(seed=self._meta_conf.seed)
        self._post_safety_check()

    def _prior_safety_check(self):
        pass

    def _post_safety_check(self):
        pass

    def initialize(self, seed: int):
        pass

    def parameter_names(self) -> List[str]:
        pass

    def loss_and_gradient(self, timer: Timer, batch: Batch, random_seed=None):
        pass

    def quality(
        self, model: torch.nn.Module, batch: Batch, metrics: Metrics
    ) -> Quality:
        pass

    def evaluate(self, dataset: Dataset) -> Quality:
        pass

    def get_auxiliary_loader(self, scenario) -> loaders.BaseLoader:
        pass

    def training_pipeline(
        self,
        meta_training_conf: Any,
        model: torch.nn.Module,
        training_loader: Type[loaders.BaseLoader],
        auxiliary_loaders: Optional[List[Type[loaders.BaseLoader]]],
        test_loader: Type[loaders.BaseLoader],
        optimizer: Type[torch.optim.Optimizer],
        scheduler: Any,
        logger: Logger,
    ) -> torch.nn.Module:
        pass

    def adapt_and_eval(
        self,
        metrics: Metrics,
        model_selection_method: Type[BaseSelection],
        current_batch: Batch,
        previous_batches: List[Batch],
        auxiliary_loaders: Optional[List[loaders.BaseLoader]],
        logger: Logger,
        timer: Timer,
    ):
        pass
