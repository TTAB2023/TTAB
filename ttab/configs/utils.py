# -*- coding: utf-8 -*-

# Functions in this file are used to deal with any work related with the configuration of datasets, models and algorithms.
import copy

from ttab.configs.algorithms import algorithm_defaults
from ttab.configs.models import model_defaults
from ttab.configs.datasets import dataset_defaults
import ttab.scenarios.define_scenario as define_scenario


def config_hparams(config, print_fn=print):
    """Populates hyperparameters with defaults implied by choices of other hyperparameters."""
    # prior safety check
    assert (
        config.model_adaptation_method is not None
    ), "model adaptation method must be specified"

    # register default arguments from default scenarios if specified by the user.
    scenario = define_scenario.get_scenario(config)
    config = define_scenario.scenario_registry(config, scenario)

    # register default arguments based on data_name.
    assert (
        config.base_data_name is not None
    ), "base_data_name of target datasets must be specified (either from default scenario, or from user-provided inputs)."
    config = defaults_registry(config, template=dataset_defaults[config.base_data_name])

    # register default arguments based on model_adaptation_method
    config = defaults_registry(
        config, template=algorithm_defaults[config.model_adaptation_method]
    )

    # regitster default arguments based on model_name and misc.
    config = defaults_registry(config, template=model_defaults[config.model_name])

    return config, scenario


def defaults_registry(config, template: dict, display_compatibility=False):
    """Populates missing (key, val) pairs in config with (key, val) in template.
    Example usage: populate config with defaults
    Args:
        - config: namespace
        - template: dict
        - display_compatibility: option to raise errors if config.key != template[key]
    """
    if template is None:
        return config

    dict_config = vars(config)
    for key, val in template.items():
        if not isinstance(val, dict):  # template[key] is non-index-able
            if key not in dict_config or dict_config[key] is None:
                dict_config[key] = val
            elif dict_config[key] != val and display_compatibility:
                raise ValueError(f"Argument {key} must be set to {val}")

        else:
            if key not in dict_config.keys():
                dict_config[key] = {}
            for kwargs_key, kwargs_val in val.items():
                if (
                    kwargs_key not in dict_config[key]
                    or dict_config[key][kwargs_key] is None
                ):
                    dict_config[key][kwargs_key] = kwargs_val
                elif (
                    dict_config[key][kwargs_key] != kwargs_val and display_compatibility
                ):
                    raise ValueError(
                        f"Argument {key}[{kwargs_key}] must be set to {kwargs_val}"
                    )
    return config


def build_dict_from_config(arg_names, config):
    dict_config = vars(config)
    return dict(
        (arg_name, dict_config[arg_name])
        for arg_name in arg_names
        if (arg_name in dict_config) and (dict_config[arg_name] is not None)
    )
