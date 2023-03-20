# -*- coding: utf-8 -*-
import copy
import functools
from typing import List, Type

import torch
import torch.nn as nn
from torch.nn import functional as F

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_adaptation.base_adaptation import BaseAdaptation
from ttab.model_selection.base_selection import BaseSelection
import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
from ttab.utils.logging import Logger
from ttab.api import Batch
from ttab.utils.timer import Timer
from ttab.utils.auxiliary import fork_rng_with_seed


class BNAdapt(BaseAdaptation):
    """
    Improving robustness against common corruptions by covariate shift adaptation,
    https://domainadaptation.org/batchnorm/

    BNAdapt leverages statistics obtained from testing dataset to adapt distribution shift.
    """

    def __init__(self, meta_conf, model):
        super().__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert (
            self._meta_conf.adapt_prior is not None
        ), "the ratio of training set statistics is required"

        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []

        for name_module, module in self._base_model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                self._adapt_module_names.append(name_module)

    def _initialize_model(self, model):
        """Configure model for use with adaptation method."""
        # disable grad.
        model.eval()
        model.requires_grad_(False)

        return model.to(self._meta_conf.device)

    def _initialize_optimizer(self, model) -> torch.optim.Optimizer:
        """No optimizer is used in BNAdapt, because it only needs to update BN layers using testing set statistics."""
        pass

    def _post_safety_check(self):

        param_grads = [
            p.requires_grad
            for p in (
                self._model.parameters()
                if not self._meta_conf.distributed
                else self._model.module.parameters()
            )
        ]
        has_any_params = any(param_grads)
        assert not has_any_params, "BNAdapt doesn't need trainable params."

        has_bn = any(
            [
                (name_module in self._adapt_module_names)
                for name_module, _ in (
                    self._model.named_modules()
                    if not self._meta_conf.distributed
                    else self._model.module.named_modules()
                )
            ]
        )
        assert has_bn, "BNAdapt needs batch normalization layers."

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        self._initialize_trainable_parameters()
        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        if len(self._adapt_module_names) > 0:
            self.adapt_setup()
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.criterion = nn.CrossEntropyLoss()
        if (
            self._meta_conf.n_train_steps > 1
        ):  # no need to do multiple-step adaptation in bn_adapt.
            self._meta_conf.n_train_steps = 1

    def construct_group_computer(self, dataset):
        """This function is used to build a new metric tracker for group-wise datasets like waterbirds."""
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.tta_loss_computer = GroupLossComputer(
            criterion=criterion,
            dataset=dataset,
            device=self._meta_conf.device,
        )
        return self.tta_loss_computer

    def _bn_swap(self, model: nn.Module, prior: float):
        """
        replace the original BN layers in the model with new defined BN layer (AdaptiveBatchNorm).
        modifying BN forward pass.
        """
        return AdaptiveBatchNorm.adapt_model(
            model, prior=prior, device=self._meta_conf.device
        )

    def reset(self):
        """recover model and optimizer to their initial states."""
        pass

    def get_auxiliary_loader(self, scenario) -> loaders.BaseLoader:
        """setup for auxiliary datasets used in test-time adaptation."""
        return self._auxiliary_data_cls.construct_auxiliary_loader(
            scenario, data_augment=True
        )

    def stochastic_restore(self):
        pass

    def compute_fishers(self, scenario, data_size):
        pass

    def one_adapt_step(
        self, model: torch.nn.Module, timer: Timer, batch: Batch, random_seed=None
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)
            loss = self.criterion(y_hat, batch._y)

        return {"loss": loss.item(), "yhat": y_hat}

    def adapt_setup(self):
        """adjust batch normalization layers."""
        self._bn_swap(self._model, self._meta_conf.adapt_prior)
        return

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

        with timer("test_adaptation"):
            log(f"\tadapt the model for {self._meta_conf.n_train_steps} steps.")
            for _ in range(self._meta_conf.n_train_steps):
                adaptation_result = self.one_adapt_step(
                    self._model, timer, current_batch, random_seed=self._meta_conf.seed
                )

        with timer("evaluate_optimal_model"):
            metrics.eval(current_batch._y, adaptation_result["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    adaptation_result["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

    @property
    def name(self):
        return "bn_adapt"


# https://github.com/bethgelab/robustness
class AdaptiveBatchNorm(nn.Module):
    """Use the source statistics as a prior on the target statistics"""

    @staticmethod
    def find_bns(parent, prior, device):
        replace_mods = []
        if parent is None:
            return []
        for name, child in parent.named_children():
            child.requires_grad_(False)
            if isinstance(child, nn.BatchNorm2d):
                module = AdaptiveBatchNorm(child, prior, device)
                replace_mods.append((parent, name, module))
            else:
                replace_mods.extend(AdaptiveBatchNorm.find_bns(child, prior, device))

        return replace_mods

    @staticmethod
    def adapt_model(model, prior, device):
        replace_mods = AdaptiveBatchNorm.find_bns(model, prior, device)
        print(f"| Found {len(replace_mods)} modules to be replaced.")
        for (parent, name, child) in replace_mods:
            setattr(parent, name, child)
        return model

    def __init__(self, layer, prior, device):
        assert prior >= 0 and prior <= 1

        super().__init__()
        self.layer = layer
        self.layer.eval()

        self.norm = nn.BatchNorm2d(
            self.layer.num_features, affine=False, momentum=1.0
        ).to(device)

        self.prior = prior

    def forward(self, input):
        self.norm(input)

        running_mean = (
            self.prior * self.layer.running_mean
            + (1 - self.prior) * self.norm.running_mean
        )
        running_var = (
            self.prior * self.layer.running_var
            + (1 - self.prior) * self.norm.running_var
        )

        return F.batch_norm(
            input,
            running_mean,
            running_var,
            self.layer.weight,
            self.layer.bias,
            False,
            0,
            self.layer.eps,
        )
