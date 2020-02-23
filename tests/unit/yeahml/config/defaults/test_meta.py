import pytest

from yeahml.config.default.default_config import DEFAULT_CONFIG
from yeahml.config.meta.parse_meta import format_meta_config


"""meta = {
    "meta": {
        # directory
        "yeahml_dir": categorical(
            default_value="yeahml", required=False
        ),  # could add a check that the location exists
        "data_name": categorical(default_value=None, required=True),
        "experiment_name": categorical(default_value=None, required=True),
        # random seed
        "rand_seed": numeric(default_value=None, required=False),
        # tracing
        "trace_level": categorical(default_value=None, required=False),
        # default path to load param information
        # TODO: this should likely move to the model config
        "default_load_params_path": categorical(
            default_value=None, required=False
        ),  # TODO: confirm path exists
    }
}"""


# I'm not sure how to best structure this
# 1. should I include errors as expected here
# 2. how do I methodically ensure each error is expected
# 3. what if I change the API?
# -- likely the different types of errors need to be broken out
ex_config = {
    # ----- REQUIRED
    # missing experiment_name
    "missing_exp_name": ({"meta": {"data_name": "jack"}}, ValueError),
    # missing data name
    "missing_data_name": ({"meta": {"experiment_name": "jack"}}, ValueError),
    # ----- bare minimum
    "bare_minimum": (
        {"meta": {"data_name": "jack", "experiment_name": "trial_01"}},
        {
            "yeahml_dir": "yeahml",
            "data_name": "jack",
            "experiment_name": "trial_01",
            "rand_seed": None,
            "trace_level": None,
            "default_load_params_path": None,
        },
    ),
    # -----
    "set_rand_seed": (
        {
            "meta": {
                # directory
                "data_name": "jack",
                "experiment_name": "trial_02",
                "rand_seed": 2,
            }
        },
        {
            "yeahml_dir": "yeahml",
            "data_name": "jack",
            "experiment_name": "trial_02",
            "rand_seed": 2,
            "trace_level": None,
            "default_load_params_path": None,
        },
    ),
    "set_rand_seed_to_float": (
        {
            "meta": {
                # directory
                "data_name": "jack",
                "experiment_name": "trial_02",
                "rand_seed": 2.2,
            }
        },
        TypeError,
    ),
    "set_rand_seed_to_float": (
        {
            "meta": {
                # directory
                "data_name": "jack",
                "experiment_name": "trial_02",
                "rand_seed": "some_string",
            }
        },
        TypeError,
    ),
    "set_data_name_to_int": (
        {
            "meta": {
                # directory
                "data_name": 3,
                "experiment_name": "trial_02",
                "rand_seed": "some_string",
            }
        },
        TypeError,
    ),
    "set_experiment_name_to_int": (
        {
            "meta": {
                # directory
                "data_name": "jack",
                "experiment_name": 3,
                "rand_seed": "some_string",
            }
        },
        TypeError,
    ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of meta"""
    if isinstance(expected, dict):
        formatted_config = format_meta_config(config["meta"], DEFAULT_CONFIG["meta"])
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = format_meta_config(
                config["meta"], DEFAULT_CONFIG["meta"]
            )
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = format_meta_config(
                config["meta"], DEFAULT_CONFIG["meta"]
            )

