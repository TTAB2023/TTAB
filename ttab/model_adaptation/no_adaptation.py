# -*- coding: utf-8 -*-
import copy
import functools
from typing import List, Type

import torch
import torch.nn as nn

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_adaptation.base_adaptation import BaseAdaptation
import ttab.model_adaptation.utils as adaptation_utils
from ttab.model_selection.base_selection import BaseSelection
from ttab.loads.define_model import load_pretrained_model
from ttab.utils.logging import Logger
from ttab.api import Batch
from ttab.utils.timer import Timer


class NoAdaptation(BaseAdaptation):
    """Standard test-time evaluation (no adaptation)."""

    def __init__(self, meta_conf, model):
        super().__init__(meta_conf, model)

    def _prior_safety_check(self):
        if not hasattr(self._meta_conf, "debug"):
            self._meta_conf.debug = False

    def convert_iabn(self, module, **kwargs):
        module_output = module
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            IABN = (
                adaptation_utils.InstanceAwareBatchNorm2d
                if isinstance(module, nn.BatchNorm2d)
                else adaptation_utils.InstanceAwareBatchNorm1d
            )
            module_output = IABN(
                num_channels=module.num_features,
                k=self._meta_conf.iabn_k,
                eps=module.eps,
                momentum=module.momentum,
                threshold=self._meta_conf.threshold_note,
                affine=module.affine,
            )

            module_output._bn = copy.deepcopy(module)

        for name, child in module.named_children():
            module_output.add_module(name, self.convert_iabn(child, **kwargs))
        del module
        return module_output

    def _initialize_model(self, model):
        if hasattr(self._meta_conf, "iabn") and self._meta_conf.iabn:
            self.convert_iabn(model)
            load_pretrained_model(self._meta_conf, model)
        model.eval()
        return model.to(self._meta_conf.device)

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))

    def construct_group_computer(self, dataset):
        """This function is used to build a new metric tracker for group-wise datasets like waterbirds."""
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.tta_loss_computer = GroupLossComputer(
            criterion=criterion,
            dataset=dataset,
            device=self._meta_conf.device,
        )
        return self.tta_loss_computer

    def reset(self):
        self._model.load_state_dict(self._base_model.state_dict())

    def quality(self, model, batch: Batch, metrics: Metrics):
        """Average quality on the batch"""
        with torch.no_grad():
            y_hat = model(batch._x)
            quality = metrics.eval(batch._y, y_hat)
        return quality

    def adapt_and_eval(
        self,
        episodic,
        metrics: Metrics,
        model_selection_method: Type[BaseSelection],
        current_batch,
        previous_batches: List[Batch],
        logger: Logger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        with timer("test_time_adaptation"):
            with torch.no_grad():
                y_hat = self._model(current_batch._x)
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

    @property
    def name(self):
        return "no_adaptation"
