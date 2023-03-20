# -*- coding: utf-8 -*-
import copy
from typing import List, Dict, Any

from ttab.api import Batch


class BaseSelection(object):
    def __init__(self, meta_conf, model):
        self.meta_conf = meta_conf
        self.model = copy.deepcopy(model).to(self.meta_conf.device)

        self.initialize()

    def initialize(self):
        pass

    def clean_up(self):
        pass

    def save_state(self):
        pass

    def select_state(
        self,
        current_batch: Batch,
        previous_batches: List[Batch],
        # auxiliary_loaders: Optional[List[loaders.BaseLoader]],
    ) -> Dict[str, Any]:
        pass
