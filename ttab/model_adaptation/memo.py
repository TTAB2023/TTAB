# -*- coding: utf-8 -*-

import copy
import time
import functools
from typing import List, Type

import torch
import torch.nn as nn

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_adaptation.base_adaptation import BaseAdaptation
import ttab.model_adaptation.utils as adaptation_utils
from ttab.model_selection.base_selection import BaseSelection
import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
from ttab.loads.datasets.cifar.data_aug_cifar import aug_cifar
from ttab.loads.datasets.imagenet.data_aug_imagenet import aug_imagenet
from ttab.loads.datasets.mnist.data_aug_mnist import aug_mnist
from ttab.utils.logging import Logger
from ttab.api import Batch
from ttab.utils.timer import Timer
from ttab.utils.auxiliary import fork_rng_with_seed


class MEMO(BaseAdaptation):
    """
    MEMO: Test Time Robustness via Adaptation and Augmentation,
    https://openreview.net/forum?id=vn74m_tWu8O

    MEMO adapts (all of) the parameters by minimizing the entropy of the model's
    average, or marginal, output distribution across the augmentations.
    """

    def __init__(self, meta_conf, model):
        super().__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert (
            self._meta_conf.aug_size > 0
        ), "The number of augmentation operation requires >= 1."
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_trainable_parameters(self, model):
        """select target parameters for adaptation methods. All of params are adapted here."""
        self._adapt_module_names = []
        params = []
        names = []

        for name_module, module in model.named_children():
            self._adapt_module_names.append(name_module)
            for name_param, param in module.named_parameters():
                params.append(param)
                names.append(f"{name_module}.{name_param}")

        return params, names

    def _initialize_model(self, model):
        """Configure model for use with adaptation method."""
        # memo works in mode eval(). All params are adapted.
        model.eval()
        model.requires_grad_(True)

        if hasattr(self._meta_conf, "bn_prior_strength") and (
            self._meta_conf.bn_prior_strength > 0
        ):
            print("modifying BN forward pass")
            nn.BatchNorm2d.prior = None
            nn.BatchNorm2d.forward = adaptation_utils.modified_bn_forward

        return model.to(self._meta_conf.device)

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""
        # particular setup of optimizer for optimal model selection.
        if self._optimal_model_selection:
            assert isinstance(
                self._meta_conf.lr_grid, list
            ), "lr_grid cannot be None in optimal model selection."
            optimizers = []
            for i in range(len(self._meta_conf.lr_grid)):
                optimizers.append(
                    adaptation_utils.define_optimizer(
                        self._meta_conf, params, lr=self._meta_conf.lr_grid[i]
                    )
                )  # list
            return optimizers
        # base case.
        optimizer = adaptation_utils.define_optimizer(
            self._meta_conf, params, lr=self._meta_conf.lr
        )
        return optimizer

    def _post_safety_check(self):

        # all modules are adapted in memo
        param_grads = [
            p.requires_grad
            for p in (
                self._model.parameters()
                if not self._meta_conf.distributed
                else self._model.module.parameters()
            )
        ]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
        assert has_all_params, "all params are trainable."

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        if self._meta_conf.model_selection_method == "optimal_model_selection":
            self._optimal_model_selection = True
            self.optimal_adaptation_steps = []
        else:
            self._optimal_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        params, names = self._initialize_trainable_parameters(model=self._model)
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        self.transform_helper = self._get_transform_helper()
        self.model_state_dict = copy.deepcopy(self._model).state_dict()

        # compute fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)

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
        """recover model and optimizer to their initial states."""
        self._model.load_state_dict(self.model_state_dict)
        if self._optimal_model_selection:
            for i in range(len(self._optimizer)):
                self._optimizer[i].load_state_dict(self._base_optimizer[i].state_dict())
        else:
            self._optimizer.load_state_dict(self._base_optimizer.state_dict())

    def stochastic_restore(self):
        """Stochastically restorre model parameters to resist catastrophic forgetting."""
        for nm, m in self._model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ["weight", "bias"] and p.requires_grad:
                    mask = (
                        (torch.rand(p.shape) < self._meta_conf.restore_prob)
                        .float()
                        .to(self._meta_conf.device)
                    )
                    with torch.no_grad():
                        p.data = self.model_state_dict[f"{nm}.{npp}"] * mask + p * (
                            1.0 - mask
                        )

    def compute_fishers(self, scenario, data_size):
        """Get fisher regularizer"""
        print("Computing fisher matrices===>")
        self.fisher_dataset, self.fisher_loader = self.get_in_test_data(
            scenario, data_size
        )

        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().to(self._meta_conf.device)
        for step, _, batch in self.fisher_loader.iterator(
            batch_size=self._meta_conf.batch_size,
            shuffle=True,
            repeat=False,
            ref_num_data=None,
            num_workers=self._meta_conf.num_workers
            if hasattr(self._meta_conf, "num_workers")
            else 2,
            pin_memory=True,
            drop_last=False,
        ):
            outputs = self._model(
                batch._x
            )  # don't need to worry about BN error becasue we use in-distribution data here.
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    if step > 1:
                        fisher = (
                            param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        )
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if step == len(self.fisher_dataset):
                        fisher = fisher / step
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            self.ewc_optimizer.zero_grad()
        print("compute fisher matrices finished")
        del self.ewc_optimizer
        self.fishers = fishers

    def get_auxiliary_loader(self, scenario) -> loaders.BaseLoader:
        """setup for auxiliary datasets used in test-time adaptation."""
        return self._auxiliary_data_cls.construct_auxiliary_loader(
            scenario, data_augment=True
        )

    def _get_transform_helper(self):
        """get particular augmentation method for different datasets"""
        if self._meta_conf.base_data_name in ["cifar10", "cifar100"]:
            return aug_cifar
        elif self._meta_conf.base_data_name in [
            "imagenet",
            "officehome",
            "waterbirds",
            "pacs",
        ]:
            return aug_imagenet
        elif self._meta_conf.base_data_name == "coloredmnist":
            return aug_mnist

    def set_nbstep_ratio(self, nbstep_ratio):
        """use a ratio to help control the number of adaptation steps"""
        assert 0 < nbstep_ratio < 1, "invalid ratio number"
        self.nbstep_ratio = nbstep_ratio

    def _get_adaptation_steps(self, index=None):
        """control the setup of adaptation step length."""
        if self._optimal_model_selection:
            return self._meta_conf.n_train_steps
        elif hasattr(self, "nbstep_ratio"):
            # last_iterate, for batch dependency experiment.
            assert (
                index is not None
            ), "the setup of per-batch adaptation step needs non-empty index"
            return max(int(self.optimal_adaptation_steps[index] * self.nbstep_ratio), 1)
        else:
            return self._meta_conf.n_train_steps

    def set_optimal_adaptation_steps(self, optimal_adaptation_steps: list):
        """set up the optimal adaptation steps as reference."""
        assert isinstance(
            optimal_adaptation_steps, list
        ), "per-batch optimal adaptation steps should be a list"
        self.optimal_adaptation_steps = optimal_adaptation_steps

    def get_optimal_adaptation_steps(self):
        "return recorded optimal adaptation steps in optimal model selection"
        assert hasattr(
            self, "optimal_adaptation_steps"
        ), "optimal_adaptation_steps is missing"
        return self.optimal_adaptation_steps

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer,
        batch: Batch,
        timer: Timer,
        random_seed=None,
    ):
        """adapt the model in one step."""
        model.eval()
        if hasattr(self._meta_conf, "bn_prior_strength") and (
            self._meta_conf.bn_prior_strength > 0
        ):
            nn.BatchNorm2d.prior = float(self._meta_conf.bn_prior_strength) / float(
                self._meta_conf.bn_prior_strength + 1
            )
        else:
            nn.BatchNorm2d.prior = 1

        with timer("forward"):
            # apply gradient accumulation.
            NUM_ACCUMULATION_STEPS = len(batch._x)
            for i in range(NUM_ACCUMULATION_STEPS):
                inputs = [
                    self.transform_helper(batch._x[i])
                    for _ in range(self._meta_conf.aug_size)
                ]
                inputs = torch.stack(inputs).to(self._meta_conf.device)

                with fork_rng_with_seed(random_seed):
                    y_hat = model(inputs)
                loss, _ = adaptation_utils.marginal_entropy(y_hat)
                loss = loss / NUM_ACCUMULATION_STEPS
                if self.fishers is not None:
                    ewc_loss = 0
                    for name, param in model.named_parameters():
                        if name in self.fishers:
                            ewc_loss += (
                                self._meta_conf.fisher_alpha
                                * (
                                    self.fishers[name][0]
                                    * (param - self.fishers[name][1]) ** 2
                                ).sum()
                            )

                loss.backward()

        with timer("update"):
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            # update parameters using accumulated gradients.
            optimizer.step()
            optimizer.zero_grad()

        nn.BatchNorm2d.prior = 1

        return {
            "optimizer": copy.deepcopy(optimizer).state_dict(),
            "loss": loss.item(),
            "grads": grads,
        }

    def run_multiple_steps(
        self,
        model,
        optimizer,
        batch,
        model_selection_method,
        nbsteps,
        timer,
        random_seed=None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                batch,
                timer,
                random_seed=random_seed,
            )
            if self._optimal_model_selection:
                model_selection_method.save_state(
                    {
                        "model": copy.deepcopy(model).state_dict()
                        if not self._meta_conf.distributed
                        else copy.deepcopy(model.module).state_dict(),
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **adaptation_result,
                    },
                    current_batch=batch,
                )
            else:
                # evaluate
                with torch.no_grad():
                    y_hat = model(batch._x)  # in eval mode
                adaptation_result["yhat"] = y_hat

                model_selection_method.save_state(
                    {
                        "model": copy.deepcopy(model).state_dict()
                        if not self._meta_conf.distributed
                        else copy.deepcopy(model.module).state_dict(),
                        "step": step,
                        "lr": self._meta_conf.lr,
                        **adaptation_result,
                    },
                    current_batch=batch,
                )

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

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                if hasattr(self._meta_conf, "bn_prior_strength") and (
                    self._meta_conf.bn_prior_strength > 0
                ):
                    nn.BatchNorm2d.prior = float(
                        self._meta_conf.bn_prior_strength
                    ) / float(self._meta_conf.bn_prior_strength + 1)
                else:
                    nn.BatchNorm2d.prior = 1

                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )
                nn.BatchNorm2d.prior = 1

        # adaptation.
        with timer("test_time_adaptation"):
            if self._optimal_model_selection:
                # (oracle) optimal model selection using ground truth label.
                assert len(current_batch._x) > 1, "batch-size should be larger than 1."
                assert (
                    self._meta_conf.lr_grid is not None
                ), "lr_grid cannot be None in optimal model selection."
                assert isinstance(
                    self._optimizer, list
                ), "optimal model selection needs a list of optimizers with varying lr."
                grid_width = len(self._optimizer)
                assert len(self._meta_conf.lr_grid) == grid_width

                # grid-search the best adaptation result per test iteration and save required information.
                current_model = copy.deepcopy(self._model)
                for i in range(grid_width):
                    self._model.load_state_dict(current_model.state_dict())
                    log(
                        f"\tadapt the model for {self._get_adaptation_steps(index=len(previous_batches))} steps with lr={self._meta_conf.lr_grid[i]}."
                    )
                    self.run_multiple_steps(
                        model=self._model,
                        optimizer=self._optimizer[i],
                        batch=current_batch,
                        model_selection_method=model_selection_method,
                        nbsteps=self._get_adaptation_steps(index=len(previous_batches)),
                        timer=timer,
                        random_seed=self._meta_conf.seed,
                    )
            else:
                # no model selection (i.e., use the last checkpoint)
                assert not isinstance(
                    self._optimizer, list
                ), "it should have only one single optimizer."
                log(
                    f"\tadapt the model for {self._get_adaptation_steps(index=len(previous_batches))} steps with lr={self._meta_conf.lr}."
                )
                self.run_multiple_steps(
                    model=self._model,
                    optimizer=self._optimizer,
                    batch=current_batch,
                    model_selection_method=model_selection_method,
                    nbsteps=self._get_adaptation_steps(index=len(previous_batches)),
                    timer=timer,
                    random_seed=self._meta_conf.seed,
                )

        # select the optimal model, and return the corresponding prediction.
        with timer("select_optimal_model"):
            # build inputs and targets for model selection.
            optimal_state = model_selection_method.select_state()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            if not self._meta_conf.distributed:
                self._model.load_state_dict(optimal_state["model"])
            else:
                self._model.module.load_state_dict(optimal_state["model"])

            model_selection_method.clean_up()

            if self._optimal_model_selection:
                # optimal model selection needs to save steps
                self.optimal_adaptation_steps.append(optimal_state["step"])
                # update optimizers.
                self._optimizer = model_selection_method.update_optimizers(
                    optimizer_state=optimal_state["optimizer"],
                    optimizer_list=self._optimizer,
                    lr_list=self._meta_conf.lr_grid,
                )

        with timer("evaluate_optimal_model"):
            metrics.eval(current_batch._y, optimal_state["yhat"])
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    optimal_state["yhat"],
                    current_batch._y,
                    current_batch._g,
                    is_training=False,
                )

        # stochastic restore.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "memo"
