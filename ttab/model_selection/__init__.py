# -*- coding: utf-8 -*-

from .last_iterate import LastIterate
from .optimal_model_selection import OptimalModelSelection


def get_model_selection_method(selection_name):
    return {
        "last_iterate": LastIterate,
        "optimal_model_selection": OptimalModelSelection,
    }[selection_name]
