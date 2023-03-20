# -*- coding: utf-8 -*-

import ttab.configs.utils as config_utils
from ttab.scenarios.default_scenarios import default_scenarios

"""define functions for scenario_registry."""

# from ttab.scenarios.default_scenarios import default_scenarios
from ttab.scenarios import (
    HomogeneousNoMixture,
    HeterogeneousNoMixture,
    InOutMixture,
    CrossMixture,
    TestCase,
    TestDomain,
    Scenario,
)
from ttab.loads.datasets.dataset_shifts import (
    data2shift,
    SyntheticShiftProperty,
    NaturalShiftProperty,
    NoShiftProperty,
    TemporalShiftProperty,
)


def get_inter_domain(config):
    assert "inter_domain" in vars(config)
    inter_domain_name = getattr(config, "inter_domain")
    inter_domain_fn = {
        "HomogeneousNoMixture": HomogeneousNoMixture,
        "HeterogeneousNoMixture": HeterogeneousNoMixture,
        "InOutMixture": InOutMixture,
        "CrossMixture": CrossMixture,
    }.get(inter_domain_name, HomogeneousNoMixture)

    if "InOutMixture" == inter_domain_name:
        arg_names = ["ratio"]
        arg_values = config_utils.build_dict_from_config(arg_names, config)
        return inter_domain_fn(**arg_values)
    elif "HeterogeneousNoMixture" == inter_domain_name:
        arg_names = ["non_iid_pattern", "non_iid_ness"]
        arg_values = config_utils.build_dict_from_config(arg_names, config)
        return inter_domain_fn(**arg_values)
    else:
        return inter_domain_fn()


def get_test_case(config):
    arg_names = [
        "data_wise",
        "batch_size",
        "offline_pre_adapt",
        "episodic",
        "intra_domain_shuffle",
    ]
    arg_values = config_utils.build_dict_from_config(arg_names, config)

    # get intra_domain/inter_domain for each test domain.
    inter_domain = get_inter_domain(config)
    return TestCase(inter_domain=inter_domain, **arg_values)


def _is_defined_name_tuple(in_object):
    return any(
        [
            isinstance(in_object, defined_named_tuple)
            for defined_named_tuple in [
                HomogeneousNoMixture,
                HeterogeneousNoMixture,
                InOutMixture,
                CrossMixture,
                TestCase,
                TestDomain,
                SyntheticShiftProperty,
                NaturalShiftProperty,
                NoShiftProperty,
                TemporalShiftProperty,
            ]
        ]
    )


def _registry_named_tuple(input):
    if _is_defined_name_tuple(input):
        new_dict = dict()
        for key, val in dict(input._asdict()).items():
            new_dict[key] = dict(val._asdict()) if _is_defined_name_tuple(val) else val
        return new_dict
    elif isinstance(input, list) and all(
        [_is_defined_name_tuple(val) for val in input]
    ):
        return [_registry_named_tuple(val) for val in input]
    else:
        return input


def scenario_registry(config, scenario):
    """This function aims to inherit arguments the scenario object and register arguments into config.
    Scenario: NamedTuple (its value may also be a NamedTuple)
    """
    # retrive name of arguments.
    field_names = list(scenario._fields)

    dict_config = vars(config)
    dict_scenario = scenario._asdict()
    for field_name in field_names:
        dict_config[field_name] = _registry_named_tuple(dict_scenario[field_name])
    return config


def extract_synthetic_info(data_name):

    # different operations to get synthetic info in various cases
    if any(
        [
            base_data_name in data_name
            for base_data_name in ["cifar10", "cifar100", "imagenet"]
        ]
    ):
        _new_data_names = data_name.split(
            "_", 2
        )  # support string like "cifar10_c_deterministic-gaussian_noise-5", "imagenet_c_deterministic-gaussian_noise-5"
        assert len(_new_data_names) == 3, "The last index indicates the shift_pattern"
        _patterns = _new_data_names[-1].split("-")
        assert len(_patterns) == 3, '<shift_state>-<shift_name>-<shift_degree>"'
        return _patterns
    elif data_name == "coloredmnist":  # TODO: not sure.
        shift_state = "stochastic"
        shift_name = "colored"
        shift_degree = 0
        return shift_state, shift_name, shift_degree


def _get_shift(config, data_name):
    # split data_name and make sure of using a correct format.
    # please check the definition of TestDomain for more details.
    _data_names = data_name.split("_")

    # extract data info.
    base_data_name = _data_names[0]
    _data_name = "_".join(_data_names[:2]) if len(_data_names) >= 2 else _data_names[0]
    shift_type = data2shift[_data_name]

    # extract shift_property.
    if shift_type == "no_shift":
        shift_property = NoShiftProperty(has_shift=False)
    elif shift_type == "natural":
        version = (
            "_".join(_data_names[2:]) if len(_data_names) > 2 else None
        )  # e.g., cifar10_shiftedlabel_constant-size-dirichlet_gaussian_noise_5
        shift_property = NaturalShiftProperty(version=version, has_shift=True)
    elif shift_type == "synthetic":
        shift_state, shift_name, shift_degree = extract_synthetic_info(data_name)
        shift_property = SyntheticShiftProperty(
            has_shift=True,
            shift_degree=int(shift_degree),
            shift_name=shift_name,
            version=shift_state,  # either 'stochastic' or 'deterministic'
        )
    elif (
        shift_type == "temporal"
    ):  # support strings like cifar10_temporal_deterministic-gaussian_noise_5. HeterogeneousNoMixture!
        version = (
            "_".join(_data_names[2:]) if len(_data_names) > 2 else None
        )  # None represents original, otherwise specify corruption type.
        shift_property = TemporalShiftProperty(version=version, has_shift=True)

    # extract domain data sampling scheme.
    arg_names = [
        "domain_sampling_name",
        "domain_sampling_value",
        "domain_sampling_ratio",
    ]
    arg_values = config_utils.build_dict_from_config(arg_names, config)
    return TestDomain(
        base_data_name=base_data_name,
        data_name=data_name,
        shift_type=shift_type,
        shift_property=shift_property,
        **arg_values,
    )


def get_scenario(config):
    # Check whether there is a specified scenario or not.
    scenario = default_scenarios.get(config.test_scenario, None)
    if scenario is not None:
        return scenario

    # Use candidate scenario determined by user rather than defaults.
    # get some basic conf.
    data_names = config.data_names.split(";")
    test_domains = [_get_shift(config, data_name) for data_name in data_names]

    # setup of test_case
    test_case = get_test_case(config)

    # init the scenario
    scenario = Scenario(
        base_data_name=config.base_data_name,
        in_data_name=config.in_data_name,
        test_domains=test_domains,
        test_case=test_case,
        task=config.task,
        model_name=config.model_name,
        model_adaptation_method=config.model_adaptation_method,
        model_selection_method=config.model_selection_method,
    )
    return scenario
