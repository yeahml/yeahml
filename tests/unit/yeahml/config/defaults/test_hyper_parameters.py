import pytest

from yeahml.config.default.create_default import DEFAULT_CONFIG
from yeahml.config.hyper_parameters.parse_hyper_parameters import (
    format_hyper_parameters_config,
)

"""
hyper_parameters = {
    "hyper_parameters": {
        "dataset": {
            "batch": numeric(default_value=None, required=True, is_type=int),
            "shuffle_buffer": numeric(default_value=None, required=False, is_type=int),
        },
        "epochs": numeric(default_value=None, required=True, is_type=int),
        # TODO: need to account for optional outter keys
        "early_stopping": optional_config(
            conf_dict={
                "epochs": numeric(default_value=None, required=False, is_type=int),
                "warm_up": numeric(default_value=None, required=False, is_type=int),
            }
        ),
        # TODO: Right now I'm assuming 1 loss and one optimizer.. this isn't
        # and needs to be reconsidered
        "optimizer": {
            "type": categorical(
                default_value=None,
                required=True,
                is_in_list=return_available_optimizers(),
            ),
            # TODO: this isn't really a list of categorical.... most are numeric
            "options": parameter_config(
                known_dict={
                    "learning_rate": numeric(
                        default_value=None, required=True, is_type=float
                    )
                }
            ),
        },
    }
}

"""

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
        {
            "hyper_parameters": {
                "dataset": {"batch": 4},
                "epochs": 2,
                "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
            }
        },
        {
            "dataset": {"batch": 4, "shuffle_buffer": None},
            "epochs": 2,
            "early_stopping": {"epochs": None, "warm_up": None},
            "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
        },
    ),
    "working_00": (
        {
            "hyper_parameters": {
                "dataset": {"batch": 3, "shuffle_buffer": 2},
                "epochs": 2,
                "early_stopping": {"epochs": 3, "warm_up": 1},
                "optimizer": {
                    "type": "adam",
                    # TODO: this isn't really a list of categorical.... most are numeric
                    "options": {"learning_rate": 0.01},
                },
            }
        },
        {
            "dataset": {"batch": 3, "shuffle_buffer": 2},
            "epochs": 2,
            "early_stopping": {"epochs": None, "warm_up": None},
            "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
        },
    ),
    "missing_epochs": (
        {
            "hyper_parameters": {
                "dataset": {"batch": 4},
                "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
            }
        },
        ValueError,
    ),
    "missing_batch": (
        {
            "hyper_parameters": {
                "dataset": None,
                "epochs": 2,
                "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
            }
        },
        ValueError,
    ),
    "fake_optimizer": (
        {
            "hyper_parameters": {
                "dataset": {"batch": 4},
                "epochs": 2,
                "optimizer": {"type": "jack", "options": {"learning_rate": 0.01}},
            }
        },
        ValueError,
    ),
    "epoch_type_float": (
        {
            "hyper_parameters": {
                "dataset": {"batch": 4},
                "epochs": 2.1,
                "optimizer": {"type": "adam", "options": {"learning_rate": 0.01}},
            }
        },
        ValueError,
    ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of performance"""
    if isinstance(expected, dict):
        formatted_config = format_hyper_parameters_config(
            config["hyper_parameters"], DEFAULT_CONFIG["hyper_parameters"]
        )
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = format_hyper_parameters_config(
                config["hyper_parameters"], DEFAULT_CONFIG["hyper_parameters"]
            )
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = format_hyper_parameters_config(
                config["hyper_parameters"], DEFAULT_CONFIG["hyper_parameters"]
            )
