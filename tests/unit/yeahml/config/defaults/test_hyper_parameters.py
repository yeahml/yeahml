import pytest

from yeahml.config.template.components.hyper_parameters import HYPER_PARAMETERS
from util import parse_default

# TODO: test options
ex_config = {
    # ----- REQUIRED
    # missing
    "missing_loss": (
        {
            "hyper_parameters": {
                "metric": {"type": ["binarycrossentropy"], "options": [None]}
            }
        },
        ValueError,
    ),
    "minimal_00": (
        {"hyper_parameters": {"dataset": {"batch": 4}, "epochs": 2}},
        {"hyper_parameters": {"dataset": {"batch": 4}, "epochs": 2}},
    ),
    "working_00": (
        {
            "hyper_parameters": {
                "dataset": {"batch": 3, "shuffle_buffer": 2},
                "epochs": 2,
                "early_stopping": {"epochs": 3, "warm_up": 1},
            }
        },
        {
            "hyper_parameters": {
                "dataset": {"batch": 3, "shuffle_buffer": 2},
                "epochs": 2,
                "early_stopping": {"epochs": 3, "warm_up": 1},
            }
        },
    ),
    "missing_epochs": ({"hyper_parameters": {"dataset": {"batch": 4}}}, ValueError),
    "missing_batch": ({"hyper_parameters": {"dataset": None, "epochs": 2}}, ValueError),
    "fake_optimizer": (
        {"hyper_parameters": {"dataset": {"batch": 4}, "epochs": 2}},
        ValueError,
    ),
    "epoch_type_float": (
        {"hyper_parameters": {"dataset": {"batch": 4}, "epochs": 2.1}},
        ValueError,
    ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of hyper parameters"""
    if isinstance(expected, dict):
        formatted_config = parse_default(config, HYPER_PARAMETERS)
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = parse_default(config, HYPER_PARAMETERS)
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = parse_default(config, HYPER_PARAMETERS)
