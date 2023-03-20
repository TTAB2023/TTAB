# -*- coding: utf-8 -*-
import copy

# from ctypes import Union
import functools
from typing import List, Optional, Type, Union

import time

import numpy as np
from scipy.spatial.distance import cdist

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


class SHOT(BaseAdaptation):
    """
    Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation
    http://proceedings.mlr.press/v119/liang20a.html

    SHOT learns the target-specific feature extraction module by exploiting both information maximization and
    self-supervised pseudo-labeling to implicitly align representations from the target domains to the source
    hypothesis.
    """

    def __init__(self, meta_conf, model):
        super(SHOT, self).__init__(meta_conf, model)

    def _prior_safety_check(self):
        # All we need to do about episodic is initializing the model at the beginning of setting up model adaptation method.
        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert (
            self._meta_conf.offline_nepoch > 0
        ), "The number of offline adaptation epoches requires >= 1"
        assert (
            self._meta_conf.auxiliary_batch_size > 0
        ), "The batch_size of auxiliary dataloaders requires >= 1"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."

    def _initialize_trainable_parameters(self):
        """During adaptation, SHOT only updates params in backbone and bottleneck. Params in classifier layer
        is freezed.
        """
        self._freezed_param_names = []
        self._adapt_module_names = []
        params = []
        names = []

        for name_module, module in self._model.named_children():
            if name_module in self._freezed_module_names:
                for name_param, _ in module.named_parameters():
                    self._freezed_param_names.append(name_param)
            else:
                self._adapt_module_names.append(name_module)
                for name_param, param in module.named_parameters():
                    params.append(param)
                    names.append(f"{name_module}.{name_param}")

        return params, names

    def _initialize_model(self, model):
        """Configure model for use with adaptation method."""
        # configure target modules for adaptation method updates: enable grad + ...
        self._freezed_module_names = ["fc", "classifier", "head"]

        model.train()
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._freezed_module_names:
                module.requires_grad_(False)

        return model.to(self._meta_conf.device)

    def _initialize_optimizer(self, params) -> torch.optim.Optimizer:
        """Set up optimizer for adaptation process."""
        # particular setup of optimizer for optimal model selection.
        if self._optimal_model_selection == True:
            assert isinstance(
                self._meta_conf.lr_grid, list
            ), "lr_grid cannot be None in optimal model selection."
            optimizers = []
            for i in range(len(self._meta_conf.lr_grid)):
                optimizer_i = adaptation_utils.define_optimizer(
                    self._meta_conf, params, lr=self._meta_conf.lr_grid[i]
                )
                for param_group in optimizer_i.param_groups:
                    param_group["lr0"] = param_group["lr"]
                optimizers.append(optimizer_i)
            return optimizers
        # base case.
        optimizer = adaptation_utils.define_optimizer(
            self._meta_conf, params, lr=self._meta_conf.lr
        )

        for param_group in optimizer.param_groups:
            param_group["lr0"] = param_group["lr"]
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
        assert not has_all_params, "not all params are trainable in shot."

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        if self._meta_conf.model_selection_method == "optimal_model_selection":
            self._optimal_model_selection = True
            if not self._meta_conf.offline_pre_adapt:
                self.optimal_adaptation_steps = []
        else:
            self._optimal_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
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

    def make_extractor_train(self, model):
        """set the extractor to training mode."""
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._adapt_module_names:
                module.train()
        return model

    def make_extractor_eval(self, model):
        """set the extractor to eval mode."""
        for index, (name_module, module) in enumerate(model.named_children()):
            if name_module in self._adapt_module_names:
                module.eval()
        return model

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
        assert (
            not self._meta_conf.offline_pre_adapt
        ), "offline shot does not need to set up optimal adaptation steps"
        self.optimal_adaptation_steps = optimal_adaptation_steps

    def get_optimal_adaptation_steps(self):
        "return recorded optimal adaptation steps in optimal model selection"
        assert hasattr(
            self, "optimal_adaptation_steps"
        ), "optimal_adaptation_steps is missing"
        assert (
            not self._meta_conf.offline_pre_adapt
        ), "offline shot does not have optimal adaptation steps"
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
        # some check
        if not hasattr(self._meta_conf, "cls_par"):
            if self._meta_conf.offline_pre_adapt:
                self._meta_conf.cls_par = 0.3
            else:
                self._meta_conf.cls_par = 0.5
        if not hasattr(self._meta_conf, "ent_par"):
            self._meta_conf.ent_par = 1.0
        assert (
            self._meta_conf.cls_par > 0 and self._meta_conf.ent_par > 0
        ), "coefficients in the objective function should be positive."

        # optimize.
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(batch._x)

            # pseudo label.
            if self._meta_conf.offline_pre_adapt:
                # offline version.
                y_label = batch._y
                classifier_loss = self._meta_conf.cls_par * nn.CrossEntropyLoss()(
                    y_hat, y_label
                )
            else:
                # online version.
                py, y_prime = F.softmax(y_hat, dim=-1).max(1)
                reliable_labels = py > self._meta_conf.threshold_shot
                classifier_loss = F.cross_entropy(
                    y_hat[reliable_labels], y_prime[reliable_labels]
                )

            # entropy loss
            entropy_loss = adaptation_utils.softmax_entropy(y_hat).mean(0)
            # divergence loss
            softmax_out = F.softmax(y_hat, dim=-1)
            msoftmax = softmax_out.mean(dim=0)
            div_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))

            loss = (
                self._meta_conf.cls_par * classifier_loss
                + self._meta_conf.ent_par * (entropy_loss + div_loss)
            )
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

    def eval_per_epoch(
        self,
        auxiliary_loader: loaders.BaseLoader,
        timer: Timer,
        logger: Logger,
        random_seed=None,
    ):
        """evaluate adapted model's performance on the auxiliary dataset."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # some init for evaluation.
        self.make_extractor_eval(self._model)
        metrics = Metrics(self._meta_conf)

        with timer("model_evaluation"):
            for batch_indx, _, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.auxiliary_batch_size,
                shuffle=False,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                pin_memory=True,
                drop_last=False,
            ):
                with torch.no_grad():
                    y_hat = self._model(batch._x)
                metrics.eval(batch._y, y_hat)
            stats = metrics.tracker()
            log(f"stats of evaluating model={stats}.")

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
                    self.make_extractor_eval(model)
                    y_hat = model(batch._x)
                    self.make_extractor_train(model)
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

    def offline_adapt(
        self,
        model_selection_method: Type[BaseSelection],
        auxiliary_loader: Optional[loaders.BaseLoader],
        timer: Timer,
        logger: Logger,
        random_seed=None,
    ):
        """implement offline adaptation given the complete dataset."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)

        # some init for model selection method.
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()
        if self._optimal_model_selection:
            # base check
            assert (
                self._meta_conf.lr_grid is not None
            ), "lr_grid cannot be None in optimal model selection."
            assert isinstance(
                self._optimizer, list
            ), "optimal model selection needs a list of optimizers with varying lr."
            grid_width = len(self._optimizer)
            assert len(self._meta_conf.lr_grid) == grid_width
        else:
            assert not isinstance(
                self._optimizer, list
            ), "it should have only one single optimizer."

        log(f"\tPrior offline adaptation begins.")
        for i in range(self._meta_conf.offline_nepoch):
            # pseudo labels are generated first and replace the labels of auxiliary loader at the beginning of each epoch.
            log(
                f"Begin generating pseudo labels for {i}-th epoch.",
                display=self._meta_conf.debug,
            )
            self.make_extractor_eval(self._model)
            ps_label = self.obtain_ps_label(
                auxiliary_loader,
                batch_size=self._meta_conf.auxiliary_batch_size,
                random_seed=random_seed + i,
            )
            ps_label = torch.from_numpy(ps_label).to(self._meta_conf.device)
            self.make_extractor_train(self._model)

            log(
                "Finished generating pseudo labels for {i}-th epoch.",
                display=self._meta_conf.debug,
            )
            num_batches = int(
                len(auxiliary_loader.dataset) / self._meta_conf.auxiliary_batch_size
            )
            drop_last = False
            if not drop_last:
                num_batches += 1

            # Use the same generator with obtain_ps_label function to control batches after shuffling.
            G = torch.Generator()
            G.manual_seed(random_seed + i)
            for batch_indx, _, batch in auxiliary_loader.iterator(
                batch_size=self._meta_conf.auxiliary_batch_size,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                generator=G,
                pin_memory=True,
                drop_last=drop_last,
            ):
                current_batch = copy.deepcopy(batch)  # for optimal model selection.
                batch._y = ps_label[
                    self._meta_conf.auxiliary_batch_size
                    * (batch_indx - 1) : self._meta_conf.auxiliary_batch_size
                    * batch_indx
                ]
                with timer("offline_adapt_model"):
                    if self._optimal_model_selection:
                        # grid-search the best adaptation result per test iteration and save required information.
                        current_model = copy.deepcopy(self._model)
                        for k in range(grid_width):
                            self._model.load_state_dict(current_model.state_dict())
                            log(
                                f"\tadapt the model for {self._get_adaptation_steps()} steps with lr={self._meta_conf.lr_grid[k]}."
                            )

                            adaptation_utils.lr_scheduler(
                                self._optimizer[k],
                                iter_ratio=(batch_indx + num_batches * i)
                                / (self._meta_conf.offline_nepoch * num_batches),
                            )

                            self.run_multiple_steps(
                                model=self._model,
                                optimizer=self._optimizer[k],
                                batch=batch,
                                model_selection_method=model_selection_method,
                                nbsteps=self._get_adaptation_steps(),
                                timer=timer,
                                random_seed=random_seed + i,
                            )
                    else:
                        # no model selection (i.e., use the last checkpoint)
                        adaptation_utils.lr_scheduler(
                            self._optimizer,
                            iter_ratio=(batch_indx + num_batches * i)
                            / (self._meta_conf.offline_nepoch * num_batches),
                        )

                        log(
                            f"\tadapt the model for {self._get_adaptation_steps()} steps with lr={self._meta_conf.lr}."
                        )
                        self.run_multiple_steps(
                            model=self._model,
                            optimizer=self._optimizer,
                            batch=batch,
                            model_selection_method=model_selection_method,
                            nbsteps=self._get_adaptation_steps(),
                            timer=timer,
                            random_seed=random_seed + i,
                        )
                # select the optimal checkpoint, and return the corresponding prediction.
                with timer("select_optimal_model"):
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
                        # update optimizers.
                        self._optimizer = model_selection_method.update_optimizers(
                            optimizer_state=optimal_state["optimizer"],
                            optimizer_list=self._optimizer,
                            lr_list=self._meta_conf.lr_grid,
                        )

            # evaluate model performance at the end of each epoch.
            log(f"\tbegin evaluating the model for the {i}-th epoch.")
            self.eval_per_epoch(
                auxiliary_loader=auxiliary_loader,
                timer=timer,
                logger=logger,
                random_seed=random_seed,
            )

    def online_adapt(
        self,
        model_selection_method: Type[BaseSelection],
        current_batch,
        previous_batches,
        logger: Logger,
        timer: Timer,
        random_seed=None,
    ):
        """implement online adaptation given the current batch of data."""
        log = functools.partial(logger.log, display=self._meta_conf.debug)
        log(f"\tinitialize selection method={model_selection_method.name}.")
        model_selection_method.initialize()

        with timer("test_time_adapt"):
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

                # grid-search the best adaptation result per test iteration.
                current_model = copy.deepcopy(self._model)
                for i in range(grid_width):
                    self._model.load_state_dict(current_model.state_dict())  # recover.
                    log(
                        f"\tadapt the model for {self._get_adaptation_steps()} steps with lr={self._meta_conf.lr_grid[i]}."
                    )
                    self.run_multiple_steps(
                        model=self._model,
                        optimizer=self._optimizer[i],
                        batch=current_batch,
                        model_selection_method=model_selection_method,
                        nbsteps=self._get_adaptation_steps(),
                        timer=timer,
                        random_seed=random_seed,
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
                    random_seed=random_seed,
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
                # online shot needs to save steps
                self.optimal_adaptation_steps.append(optimal_state["step"])
                # update optimizers.
                self._optimizer = model_selection_method.update_optimizers(
                    optimizer_state=optimal_state["optimizer"],
                    optimizer_list=self._optimizer,
                    lr_list=self._meta_conf.lr_grid,
                )

        return optimal_state["yhat"]

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
        if not self._meta_conf.offline_pre_adapt and episodic:
            log("\treset model to initial state during the test time.")
            self.reset()

        if self._meta_conf.offline_pre_adapt:
            # offline mode.
            self.make_extractor_eval(self._model)
            with torch.no_grad():
                y_hat = self._model(current_batch._x)
        else:
            if self._meta_conf.record_preadapted_perf:
                with timer("evaluate_preadapted_performance"):
                    self._model.eval()
                    with torch.no_grad():
                        yhat = self._model(current_batch._x)
                    self._model.train()
                    metrics.eval_auxiliary_metric(
                        current_batch._y, yhat, metric_name="preadapted_accuracy_top1"
                    )

            # online mode.
            y_hat = self.online_adapt(
                model_selection_method=model_selection_method,
                current_batch=current_batch,
                previous_batches=previous_batches,
                logger=logger,
                timer=timer,
                random_seed=self._meta_conf.seed,
            )

        with timer("evaluate_adapted_model"):
            metrics.eval(current_batch._y, y_hat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    y_hat, current_batch._y, current_batch._g, is_training=False
                )

        # stochastic restore.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    def obtain_ps_label(
        self,
        dataset_loader: Union[loaders.NormalLoader, loaders.BaseLoader],
        batch_size: int,  # param used to define the batch property of a dataset iterator.
        random_seed: None,
    ):
        """
        Apply a self-supervised pseudo-labeling method for each unlabeled data to better supervise
        the target encoding training.

        https://github.com/tim-learn/SHOT/blob/master/digit/uda_digit.py
        """
        if not hasattr(self._meta_conf, "threshold_shot"):
            self._meta_conf.threshold_shot = 0

        start_test = True
        with torch.no_grad():
            G = torch.Generator()
            G.manual_seed(random_seed)
            for step, _, batch in dataset_loader.iterator(
                batch_size=batch_size,
                shuffle=True,
                repeat=False,
                ref_num_data=None,
                num_workers=self._meta_conf.num_workers
                if hasattr(self._meta_conf, "num_workers")
                else 2,
                generator=G,
                pin_memory=True,
                drop_last=False,
            ):
                inputs = batch._x
                labels = batch._y
                children_modules = []
                for name, module in self._model.named_children():
                    if name in self._adapt_module_names:
                        children_modules.append(module)

                feas = inputs
                for i in range(len(children_modules)):
                    feas = children_modules[i](feas)
                feas = feas.view(feas.size(0), -1)

                outputs = self._model(inputs)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        # accuracy = torch.sum(torch.squeeze(predict.to(self._meta_conf.device)).float() == all_label).item() / float(all_label.size()[0])

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count > self._meta_conf.threshold_shot)
        labelset = labelset[0]

        # dd = cdist(all_fea, initc, 'cosine')
        dd = cdist(all_fea, initc[labelset], "cosine")
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
        # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            # dd = cdist(all_fea, initc, 'cosine')
            dd = cdist(all_fea, initc[labelset], "cosine")
            pred_label = dd.argmin(axis=1)
            pred_label = labelset[pred_label]

        return pred_label.astype("int")

    @property
    def name(self):
        return "shot"
