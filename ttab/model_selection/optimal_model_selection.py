# -*- coding: utf-8 -*-
import copy
from typing import List, Dict, Any

import torch

from ttab.api import Batch
from ttab.model_selection.base_selection import BaseSelection
from ttab.model_selection.metrics import accuracy_top1, cross_entropy


class OptimalModelSelection(BaseSelection):
    """grid-search the best adaptation result per test iteration (given a sufficiently long adaptation
    steps and then iterate over different learning rates, save the best model with a certain
    learning rate after running a certain number of steps and its optimizer states)"""

    def __init__(self, meta_conf, model):
        super().__init__(meta_conf, model)

    def initialize(self):
        if hasattr(self.model, "ssh"):
            self.model.ssh.eval()
            self.model.main_model.eval()
        else:
            self.model.eval()

        self.optimal_state = None
        self.current_batch_best_acc = 0
        self.current_batch_coupled_ent = None

    def clean_up(self):
        self.optimal_state = None
        self.current_batch_best_acc = 0
        self.current_batch_coupled_ent = None

    def save_state(self, state, current_batch):
        """Selectively save state for current batch of data."""
        batch_best_acc = self.current_batch_best_acc
        coupled_ent = self.current_batch_coupled_ent

        if not hasattr(self.model, "ssh"):
            self.model.load_state_dict(state["model"])
            with torch.no_grad():
                outputs = self.model(current_batch._x)
        else:
            self.model.main_model.load_state_dict(state["main_model"])
            with torch.no_grad():
                outputs = self.model.main_model(current_batch._x)

        current_acc = self.cal_acc(current_batch._y, outputs)
        if (self.optimal_state is None) or (current_acc > batch_best_acc):
            self.current_batch_best_acc = current_acc
            self.current_batch_coupled_ent = self.cal_ent(current_batch._y, outputs)
            state["yhat"] = outputs
            self.optimal_state = state
        elif current_acc == batch_best_acc:
            # compare cross entropy
            assert coupled_ent is not None, "Cross entropy value cannot be none."
            current_ent = self.cal_ent(current_batch._y, outputs)
            if current_ent < coupled_ent:
                self.current_batch_coupled_ent = current_ent
                state["yhat"] = outputs
                self.optimal_state = state

    def cal_acc(self, targets, outputs):
        return accuracy_top1(targets, outputs)

    def cal_ent(self, targets, outputs):
        return cross_entropy(targets, outputs)

    def select_state(self) -> Dict[str, Any]:
        """return the optimal state and sync the model defined in the model selection method."""
        if not hasattr(self.model, "ssh"):
            self.model.load_state_dict(self.optimal_state["model"])
        else:
            self.model.main_model.load_state_dict(self.optimal_state["main_model"])
            self.model.ssh.load_state_dict(self.optimal_state["ssh"])
        return self.optimal_state

    def update_optimizers(self, optimizer_state, optimizer_list, lr_list):
        for i in range(len(optimizer_list)):
            optimizer_list[i].load_state_dict(optimizer_state)
            for param_group in optimizer_list[i].param_groups:
                param_group["lr"] = lr_list[i]
        return optimizer_list

    def replace_model_arch(self, model):
        self.model = copy.deepcopy(model)

    @property
    def name(self):
        return "optimal_model_selection"
