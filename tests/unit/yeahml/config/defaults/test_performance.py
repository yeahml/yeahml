import pytest

from yeahml.config.template.components.performance import PERFORMANCE
from util import parse_default

# TODO: test options
ex_config = {
    # ----- REQUIRED
    # missing metric
    # "missing_metric": (
    #     {
    #         "performance": {
    #             "main": {"loss": {"type": "binary_crossentropy", "options": None}}
    #         }
    #     },
    #     ValueError,
    # ),
    # "missing_loss": (
    #     {
    #         "performance": {
    #             "main": {"metric": {"type": ["binarycrossentropy"], "options": [None]}}
    #         }
    #     },
    #     ValueError,
    # ),
    "working_00": (
        {
            "performance": {
                "objectives": {
                    "main": {
                        "metric": {"type": "binarycrossentropy", "options": [None]},
                        "loss": {
                            "type": "binary_crossentropy",
                            "options": None,
                            "track": "mean",
                        },
                        "in_config": {
                            "type": "supervised",
                            "options": {"prediction": "out", "target": "some_target"},
                            "dataset": "some_dataset",
                        },
                    }
                }
            }
        },
        {
            "performance": {
                "objectives": {
                    "main": {
                        "metric": {"type": ["binarycrossentropy"], "options": [None]},
                        "loss": {
                            "type": "binary_crossentropy",
                            "options": [None],
                            "track": ["mean"],
                        },
                        "in_config": {
                            "type": "supervised",
                            "options": {"prediction": "out", "target": "some_target"},
                            "dataset": "some_dataset",
                        },
                    }
                }
            }
        },
    ),
    # "loss_type_as_lists": (
    #     {
    #         "performance": {
    #             "main": {
    #                 "metric": {"type": ["binarycrossentropy"], "options": [None]},
    #                 "loss": {"type": ["binary_crossentropy"], "options": None},
    #             }
    #         }
    #     },
    #     TypeError,
    # ),
    # "loss_options_as_lists": (
    #     {
    #         "performance": {
    #             "main": {
    #                 "metric": {"type": ["binarycrossentropy"], "options": [None]},
    #                 "loss": {"type": "binary_crossentropy", "options": [None]},
    #             }
    #         }
    #     },
    #     TypeError,
    # ),
    # this seems to work for either a ValueError or type error.. but I only care
    # about the value error.. unsure
    # "not_available_metric_type": (
    #     {
    #         "performance": {
    #             "main": {
    #                 "metric": {"type": ["rmse"], "options": [None]},
    #                 "loss": {"type": "binary_crossentropy", "options": None},
    #             }
    #         }
    #     },
    #     ValueError,
    # ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of performance"""
    if isinstance(expected, dict):
        formatted_config = parse_default(config, PERFORMANCE)
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = parse_default(config, PERFORMANCE)
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = parse_default(config, PERFORMANCE)
