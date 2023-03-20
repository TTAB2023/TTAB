# -*- coding: utf-8 -*-
import copy
import functools
import time
from typing import List, Type
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_adaptation.base_adaptation import BaseAdaptation
import ttab.model_adaptation.utils as adaptation_utils
from ttab.model_selection.base_selection import BaseSelection
import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
from ttab.utils.logging import Logger
from ttab.api import Batch
from ttab.utils.timer import Timer
from ttab.utils.auxiliary import fork_rng_with_seed


class ConjugatePL(BaseAdaptation):
    """Test-time Adaptation via Conjugate Pseudo-Labels,
    https://arxiv.org/abs/2207.09640
    """

    def __init__(self, meta_conf, model):
        super(ConjugatePL, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_trainable_parameters(self):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        self._adapt_module_names = []
        params = []
        names = []

        for name_module, module in self._model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets
            if "layer4" in name_module:
                continue
            # if isinstance(self._model, ResNetCifar) and 'layer3' in name_module:
            #     continue
            if isinstance(module, nn.BatchNorm2d):
                self._adapt_module_names.append(name_module)
                for name_parameter, parameter in module.named_parameters():
                    if name_parameter in ["weight", "bias"]:
                        params.append(parameter)
                        names.append(f"{name_module}.{name_parameter}")

        assert (
            len(self._adapt_module_names) > 0
        ), "Tent needs batch normalization layers."
        return params, names

    def _initialize_model(self, model):
        """Configure model for use with adaptation method."""
        model.train()
        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        # configure target modules for adaptation method updates: enable grad + ...
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # bn module always uses batch statistics, in both training and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        return model.to(self._meta_conf.device)

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""
        # particular setup of optimizer for optimal model selection.
        base_optimizer = torch.optim.SGD
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
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

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
        assert not has_all_params, "not all params are trainable."

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
        assert has_bn, "The adaptation method needs batch normalization layers."

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        if self._meta_conf.model_selection_method == "optimal_model_selection":
            self._optimal_model_selection = True
            self.optimal_adaptation_steps = []
        else:
            self._optimal_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model)  # update base model
        params, names = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )
        # compute fisher regularizer
        self.fishers = None
        self.ewc_optimizer = torch.optim.SGD(params, 0.001)

        # base model state.
        self.model_state_dict = copy.deepcopy(self._model).state_dict()

    def construct_group_computer(self, dataset):
        """This function is used to build a new metric tracker for group-wise datasets like waterbirds."""
        criterion = nn.CrossEntropyLoss(reduction="none")
        self.tta_loss_computer = GroupLossComputer(
            criterion=criterion,
            dataset=dataset,
            device=self._meta_conf.device,
        )
        return self.tta_loss_computer

    def make_model_parallel(self):
        assert torch.cuda.device_count() > 1, "no enough available devices."
        self._model = torch.nn.DataParallel(self._model)

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

    def get_in_test_data(self, scenario, data_size):
        return self._auxiliary_data_cls.construct_in_dataset(
            scenario, data_size, data_augment=True
        )

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
        optimizer.zero_grad()
        eps = self._meta_conf.model_eps
        num_classes = self._meta_conf.statistics["n_classes"]
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)
            y_hat = y_hat / self._meta_conf.temperature_scaling
            softmax_prob = F.softmax(y_hat, dim=1)

            # adapt.
            smax_inp = softmax_prob
            eye = torch.eye(num_classes).to(y_hat.device)
            eye = eye.reshape((1, num_classes, num_classes))
            eye = eye.repeat(y_hat.shape[0], 1, 1)
            t2 = eps * torch.diag_embed(smax_inp)
            smax_inp = torch.unsqueeze(smax_inp, 2)
            t3 = eps * torch.bmm(smax_inp, torch.transpose(smax_inp, 1, 2))
            matrix = eye + t2 - t3
            # y_star = torch.linalg.solve(matrix, smax_inp)  # only support in a latest torch version.
            y_star, _ = torch.solve(smax_inp, matrix)
            y_star = torch.squeeze(y_star)

            pseudo_prob = y_star
            loss = torch.logsumexp(y_hat, dim=1) - (
                pseudo_prob * y_hat - eps * pseudo_prob * (1 - softmax_prob)
            ).sum(dim=1)
            loss = loss.mean()

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
                loss += ewc_loss

        with timer("backward"):
            loss.backward()
            grads = dict(
                (name, param.grad.clone().detach())
                for name, param in model.named_parameters()
                if param.grad is not None
            )
            optimizer.step()
            optimizer.zero_grad()
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
                model.eval()
                with torch.no_grad():
                    y_hat = model(batch._x)
                adaptation_result["yhat"] = y_hat
                model.train()

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
        # group_logger:CSVBatchLogger,
        timer: Timer,
    ):
        """The key entry of test-time adaptation."""
        # some simple initialization.
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        if episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        if self._optimal_model_selection and len(previous_batches) == 0:
            model_selection_method.replace_model_arch(model=self._model)

        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        if self._meta_conf.record_preadapted_perf:
            with timer("evaluate_preadapted_performance"):
                self._model.eval()
                with torch.no_grad():
                    yhat = self._model(current_batch._x)
                self._model.train()
                metrics.eval_auxiliary_metric(
                    current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                )

        # adaptation.
        with timer("test_time_adaptation"):
            if self._optimal_model_selection:
                # (oracle) optimal model selection using ground truth label.
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
                    self._model.load_state_dict(current_model.state_dict())  # recover.
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

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
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
                # log after each batch. Uncomment it if need.
                # group_logger.log(0, len(previous_batches), self.tta_loss_computer.get_stats())
                # group_logger.flush()

        # stochastic restore.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "conjugate_pl"
