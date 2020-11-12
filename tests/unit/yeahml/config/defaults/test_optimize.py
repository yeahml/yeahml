import pytest

from yeahml.config.template.components.optimize import OPTIMIZE
from util import parse_default

ex_config = {
    # ----- REQUIRED
    # missing
    "working_00": (
        {
            "optimize": {
                "optimizers": {
                    "main_opt": {
                        "type": "adam",
                        "options": {"learning_rate": 0.0001},
                        "objectives": ["main_opt"],
                    }
                },
                # "directive": {"instructions": "main_opt"},
            }
        },
        {
            "optimize": {
                "optimizers": {
                    "main_opt": {
                        "type": "adam",
                        "options": {"learning_rate": 0.0001},
                        "objectives": ["main_opt"],
                    }
                },
                # "directive": {
                #     "instructions": {
                #         "YEAHML_0": {"operation": "+", "optimizers": "main_opt"}
                #     }
                # },
            }
        },
    )
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of optimize"""
    if isinstance(expected, dict):
        formatted_config = parse_default(config, OPTIMIZE)
        print(formatted_config)
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = parse_default(config, OPTIMIZE)
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = parse_default(config, OPTIMIZE)
