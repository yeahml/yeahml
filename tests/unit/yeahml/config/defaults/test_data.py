import pytest

from yeahml.config.template.components.data import DATA
from util import parse_default


# TODO: test options
# NOTE: multiple of the same name would need to be caught before here since the
# type is a dict and the same name will overwrite the current value
ex_config = {
    # ----- REQUIRED
    # missing
    "minimal_00": (
        {
            "data": {
                "datasets": {
                    "mnist": {
                        "in": {"in_a": {"shape": [64, 64], "dtype": "float64"}},
                        "split": {"names": ["train", "val"]},
                    }
                }
            }
        },
        {
            "data": {
                "datasets": {
                    "mnist": {
                        "in": {
                            "in_a": {
                                "shape": [64, 64],
                                "dtype": "float64",
                                "startpoint": True,
                                "endpoint": False,
                                "label": False,
                            }
                        },
                        "split": {"names": ["train", "val"]},
                    }
                }
            }
        },
    ),
    "shape_is_None": (
        {"data": {"in": {"in_a": {"shape": [None], "dtype": "float64"}}}},
        ValueError,
    ),
    "dtype_is_None": (
        {"data": {"in": {"in_a": {"shape": [64, 64], "dtype": None}}}},
        ValueError,
    ),
    "shape_incorrect_type": (
        {"data": {"in": {"in_a": {"shape": "hello", "dtype": "float64"}}}},
        TypeError,
    ),
    "dtype_incorrect_type": (
        {"data": {"in": {"in_a": {"shape": [64, 64], "dtype": 0.33}}}},
        TypeError,
    ),
    "dtype_not_exist": (
        {"data": {"in": {"in_a": {"shape": [64, 64], "dtype": "made_up_dtype"}}}},
        ValueError,
    ),
    # TODO: allow this -- where a None is present in the list
    # "minimal_00": (
    #     {"data": {"in": {"in_a": {"shape": [None, 64], "dtype": "float64"}}}},
    #     {"in": {"in_a": {"dtype": "float64", "shape": [64, 64]}}},
    # ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of data"""
    if isinstance(expected, dict):
        formatted_config = parse_default(config, DATA)
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = parse_default(config, DATA)
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = parse_default(config, DATA)
