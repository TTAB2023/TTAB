# -*- coding: utf-8 -*-
import copy
import functools
import time
from typing import List, Type
import random

import torch
import torch.nn as nn

from ttab.model_selection.metrics import Metrics
from ttab.model_selection.group_metrics import GroupLossComputer
from ttab.model_adaptation.base_adaptation import BaseAdaptation
import ttab.model_adaptation.utils as adaptation_utils
from ttab.model_selection.base_selection import BaseSelection
import ttab.loads.datasets.loaders as loaders
import ttab.loads.define_dataset as define_dataset
from ttab.loads.define_model import load_pretrained_model
from ttab.utils.logging import Logger
from ttab.api import Batch
from ttab.utils.timer import Timer
from ttab.utils.auxiliary import fork_rng_with_seed


class NOTE(BaseAdaptation):
    """Tent: Fully Test-Time Adaptation by Entropy Minimization,
    https://openreview.net/forum?id=uXl3bZLkr3c

    Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """

    def __init__(self, meta_conf, model):
        super(NOTE, self).__init__(meta_conf, model)

    def _prior_safety_check(self):

        assert (
            self._meta_conf.debug is not None
        ), "The state of debug should be specified"
        assert self._meta_conf.n_train_steps > 0, "adaptation steps requires >= 1."
        assert self._meta_conf.use_learned_stats, "NOTE uses batch-free evaluation."

    def _initialize_trainable_parameters(self):
        """select target params for adaptation methods."""
        self._adapt_module_names = []
        params = []
        names = []

        for name_module, module in self._model.named_children():
            self._adapt_module_names.append(name_module)
            for name_param, param in module.named_parameters():
                params.append(param)
                names.append(f"{name_module}.{name_param}")

        return params, names

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
        """Configure model for use with adaptation method."""
        # IABN
        if self._meta_conf.iabn:
            # check BN layers
            bn_flag = False
            for name_module, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_flag = True
            assert bn_flag, "IABN needs batch normalization layers."
            self.convert_iabn(model)
            load_pretrained_model(self._meta_conf, model)

        # disable grad, to (re-)enable only what specified adaptation method updates
        model.requires_grad_(False)
        # configure target modules for adaptation method updates: enable grad + ...
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                if self._meta_conf.use_learned_stats:
                    module.track_running_stats = True
                    module.momentum = self._meta_conf.bn_momentum
                else:
                    # with below, this module always uses the test batch statistics (no momentum)
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None

                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            elif isinstance(module, nn.InstanceNorm1d) or isinstance(
                module, nn.InstanceNorm2d
            ):  # ablation study
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

            if self._meta_conf.iabn:
                if isinstance(
                    module, adaptation_utils.InstanceAwareBatchNorm2d
                ) or isinstance(module, adaptation_utils.InstanceAwareBatchNorm1d):
                    for param in module.parameters():
                        param.requires_grad = True

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
        is_training = self._model.training
        assert is_training, "adaptation needs train mode: call model.train()."

        param_grads = [p.requires_grad for p in (self._model.parameters())]
        has_any_params = any(param_grads)
        has_all_params = all(param_grads)
        assert has_any_params, "adaptation needs some trainable params."
        assert not has_all_params, "not all params are trainable."

    def initialize(self, seed: int):
        """Initialize the benchmark."""
        if self._meta_conf.model_selection_method == "optimal_model_selection":
            self._optimal_model_selection = True
            self.optimal_adaptation_steps = []
        else:
            self._optimal_model_selection = False

        self._model = self._initialize_model(model=copy.deepcopy(self._base_model))
        self._base_model = copy.deepcopy(self._model)  # update base model
        self.fifo = FIFO(capacity=self._meta_conf.update_every_x)
        self.memory = self.define_memory()
        params, _ = self._initialize_trainable_parameters()
        self._optimizer = self._initialize_optimizer(params)
        self.entropy_loss = adaptation_utils.HLoss(
            temp_factor=self._meta_conf.temperature
        )
        self._base_optimizer = copy.deepcopy(self._optimizer)
        self._auxiliary_data_cls = define_dataset.ConstructAuxiliaryDataset(
            config=self._meta_conf
        )

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

    def define_memory(self):
        if self._meta_conf.memory_type == "FIFO":
            mem = FIFO(capacity=self._meta_conf.memory_size)
        elif self._meta_conf.memory_type == "Reservoir":
            mem = Reservoir(capacity=self._meta_conf.memory_size)
        elif self._meta_conf.memory_type == "PBRS":
            mem = PBRS(
                capacity=self._meta_conf.memory_size,
                num_class=self._meta_conf.statistics["n_classes"],
            )

        return mem

    def update_memory(self, current_batch):
        for i in range(len(current_batch)):
            current_sample = current_batch[i]
            self.fifo.add_instance(current_sample)
            with torch.no_grad():
                self._model.eval()
                if self._meta_conf.memory_type in ["FIFO", "Reservoir"]:
                    self.memory.add_instance(current_sample)
                elif self._meta_conf.memory_type in ["PBRS"]:
                    f, c = current_sample[0].to(self._meta_conf.device), current_sample[
                        1
                    ].to(self._meta_conf.device)

                    logit = self._model(f.unsqueeze(0))
                    pseudo_cls = logit.max(1, keepdim=False)[1][0]
                    self.memory.add_instance([f, pseudo_cls, c, 0])

    def one_adapt_step(
        self,
        model: torch.nn.Module,
        optimizer,
        memory_sampled_feats,
        timer: Timer,
        random_seed=None,
    ):
        """adapt the model in one step."""
        with timer("forward"):
            with fork_rng_with_seed(random_seed):
                y_hat = model(memory_sampled_feats)
            loss = self.entropy_loss(y_hat)

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
        memory_sampled_feats,
        current_batch,
        model_selection_method,
        nbsteps,
        timer,
        random_seed=None,
    ):
        for step in range(1, nbsteps + 1):
            adaptation_result = self.one_adapt_step(
                model,
                optimizer,
                memory_sampled_feats,
                timer,
                random_seed=random_seed,
            )
            if self._optimal_model_selection:
                model_selection_method.save_state(
                    {
                        "model": copy.deepcopy(model).state_dict(),
                        "step": step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **adaptation_result,
                    },
                    current_batch=current_batch,
                )
            else:
                model_selection_method.save_state(
                    {
                        "model": copy.deepcopy(model).state_dict(),
                        "step": step,
                        "lr": self._meta_conf.lr,
                        **adaptation_result,
                    },
                    current_batch=current_batch,
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

        # use new samples to update memory
        self.update_memory(current_batch)
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

                # Test-time adapt
                self._model.train()
                (
                    memory_sampled_feats,
                    _,
                ) = self.memory.get_memory()  # get pseudo iid batch
                memory_sampled_feats = torch.stack(memory_sampled_feats)
                memory_sampled_feats = memory_sampled_feats.to(self._meta_conf.device)
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
                        memory_sampled_feats=memory_sampled_feats,
                        current_batch=current_batch,
                        model_selection_method=model_selection_method,
                        nbsteps=self._get_adaptation_steps(index=len(previous_batches)),
                        timer=timer,
                        random_seed=self._meta_conf.seed,
                    )
            else:
                # no model selection (i.e., use the last checkpoint)
                with torch.no_grad():
                    self._model.eval()
                    yhat = self._model(current_batch._x)
                # metrics.eval(current_batch._y, yhat)

                # Test-time adapt
                self._model.train()
                (
                    memory_sampled_feats,
                    _,
                ) = self.memory.get_memory()  # get pseudo iid batch
                memory_sampled_feats = torch.stack(memory_sampled_feats)
                memory_sampled_feats = memory_sampled_feats.to(self._meta_conf.device)
                self.run_multiple_steps(
                    model=self._model,
                    optimizer=self._optimizer,
                    memory_sampled_feats=memory_sampled_feats,
                    current_batch=current_batch,
                    model_selection_method=model_selection_method,
                    nbsteps=self._get_adaptation_steps(index=len(previous_batches)),
                    timer=timer,
                    random_seed=self._meta_conf.seed,
                )

        # select the optimal checkpoint, and return the corresponding prediction.
        with timer("select_optimal_checkpoint"):
            optimal_state = model_selection_method.select_state()
            self._model.load_state_dict(optimal_state["model"])
            model_selection_method.clean_up()
            log(
                f"\tselect the optimal model ({optimal_state['step']}-th step and lr={optimal_state['lr']}) for the current mini-batch.",
            )

            if self._optimal_model_selection:
                yhat = optimal_state["yhat"]
                # optimal model selection needs to save steps
                self.optimal_adaptation_steps.append(optimal_state["step"])
                # update optimizers.
                self._optimizer = model_selection_method.update_optimizers(
                    optimizer_state=optimal_state["optimizer"],
                    optimizer_list=self._optimizer,
                    lr_list=self._meta_conf.lr_grid,
                )

        with timer("evaluate_optimal_model"):
            metrics.eval(current_batch._y, yhat)
            if self._meta_conf.base_data_name in ["waterbirds"]:
                self.tta_loss_computer.loss(
                    yhat, current_batch._y, current_batch._g, is_training=False
                )
                # log after each batch. Uncomment it if need.
                # group_logger.log(0, len(previous_batches), self.tta_loss_computer.get_stats())
                # group_logger.flush()

        # stochastic restore.
        if self._meta_conf.stochastic_restore_model:
            self.stochastic_restore()

    @property
    def name(self):
        return "note"


class FIFO:
    def __init__(self, capacity):
        self.data = [[], []]
        self.capacity = capacity
        pass

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert len(instance) == 2

        if self.get_occupancy() >= self.capacity:
            self.remove_instance()

        for i, dim in enumerate(self.data):
            dim.append(instance[i])

    def remove_instance(self):
        for dim in self.data:
            dim.pop(0)
        pass


class Reservoir:  # Time uniform
    def __init__(self, capacity):
        super(Reservoir, self).__init__(capacity)
        self.data = [[], []]
        self.capacity = capacity
        self.counter = 0

    def get_memory(self):
        return self.data

    def get_occupancy(self):
        return len(self.data[0])

    def add_instance(self, instance):
        assert len(instance) == 2
        is_add = True
        self.counter += 1

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance()

        if is_add:
            for i, dim in enumerate(self.data):
                dim.append(instance[i])

    def remove_instance(self):

        m = self.get_occupancy()
        n = self.counter
        u = random.uniform(0, 1)
        if u <= m / n:
            tgt_idx = random.randrange(0, m)  # target index to remove
            for dim in self.data:
                dim.pop(tgt_idx)
        else:
            return False
        return True


class PBRS:
    def __init__(self, capacity, num_class):
        self.data = [
            [[], []] for _ in range(num_class)
        ]  # feat, pseudo_cls, domain, cls, loss
        self.counter = [0] * num_class
        self.marker = [""] * num_class
        self.num_class = num_class
        self.capacity = capacity
        pass

    def print_class_dist(self):

        print(self.get_occupancy_per_class())

    def print_real_class_dist(self):

        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            for cls in data_per_cls[2]:
                occupancy_per_class[cls] += 1
        print(occupancy_per_class)

    def get_memory(self):

        data = self.data

        tmp_data = [[], []]
        for data_per_cls in data:
            feats, cls = data_per_cls
            tmp_data[0].extend(feats)
            tmp_data[1].extend(cls)

        return tmp_data

    def get_occupancy(self):
        occupancy = 0
        for data_per_cls in self.data:
            occupancy += len(data_per_cls[0])
        return occupancy

    def get_occupancy_per_class(self):
        occupancy_per_class = [0] * self.num_class
        for i, data_per_cls in enumerate(self.data):
            occupancy_per_class[i] = len(data_per_cls[0])
        return occupancy_per_class

    def update_loss(self, loss_list):
        for data_per_cls in self.data:
            feats, cls, dls, _, losses = data_per_cls
            for i in range(len(losses)):
                losses[i] = loss_list.pop(0)

    def add_instance(self, instance):
        assert len(instance) == 4
        cls = instance[1]
        self.counter[cls] += 1
        is_add = True

        if self.get_occupancy() >= self.capacity:
            is_add = self.remove_instance(cls)

        if is_add:
            for i, dim in enumerate(self.data[cls]):
                dim.append(instance[i])

    def get_largest_indices(self):

        occupancy_per_class = self.get_occupancy_per_class()
        max_value = max(occupancy_per_class)
        largest_indices = []
        for i, oc in enumerate(occupancy_per_class):
            if oc == max_value:
                largest_indices.append(i)
        return largest_indices

    def remove_instance(self, cls):
        largest_indices = self.get_largest_indices()
        if (
            cls not in largest_indices
        ):  #  instance is stored in the place of another instance that belongs to the largest class
            largest = random.choice(largest_indices)  # select only one largest class
            tgt_idx = random.randrange(
                0, len(self.data[largest][0])
            )  # target index to remove
            for dim in self.data[largest]:
                dim.pop(tgt_idx)
        else:  # replaces a randomly selected stored instance of the same class
            m_c = self.get_occupancy_per_class()[cls]
            n_c = self.counter[cls]
            u = random.uniform(0, 1)
            if u <= m_c / n_c:
                tgt_idx = random.randrange(
                    0, len(self.data[cls][0])
                )  # target index to remove
                for dim in self.data[cls]:
                    dim.pop(tgt_idx)
            else:
                return False
        return True
