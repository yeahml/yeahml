import pytest

from yeahml.config.default.default_config import DEFAULT_CONFIG
from yeahml.config.logging.parse_logging import format_logging_config

"""
logging = {
    "logging": {
        "console": {
            "level": categorical(
                default_value="CRITICAL",
                required=False,
                is_in_list=ERR_LEVELS,
                is_type=str,
            ),
            "format_str": categorical(
                default_value="%(name)-12s: %(levelname)-8s %(message)s",
                required=False,
                is_type=str,
            ),
        },
        "file": {
            "level": categorical(
                default_value="CRITICAL",
                required=False,
                is_in_list=ERR_LEVELS,
                is_type=str,
            ),
            "format_str": categorical(
                default_value="%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s",
                required=False,
                is_type=str,
            ),
        },
    }
}

"""

# TODO: test options
ex_config = {
    # ----- REQUIRED
    # missing
    "minimal_00": (
        {"logging": {}},
        {
            "console": {
                "level": "critical",
                "format_str": "%(name)-12s: %(levelname)-8s %(message)s",
            },
            "file": {
                "level": "critical",
                "format_str": "%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s",
            },
        },
    ),
    "working_00": (
        {
            "logging": {
                "console": {
                    "level": "critical",
                    "format_str": "%(name)-12s: %(levelname)-8s %(message)s",
                },
                "file": {
                    "level": "critical",
                    "format_str": "%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s",
                },
            }
        },
        {
            "console": {
                "level": "critical",
                "format_str": "%(name)-12s: %(levelname)-8s %(message)s",
            },
            "file": {
                "level": "critical",
                "format_str": "%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s",
            },
        },
    ),
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of performance"""
    if isinstance(expected, dict):
        formatted_config = format_logging_config(
            config["logging"], DEFAULT_CONFIG["logging"]
        )
        assert expected == formatted_config
    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = format_logging_config(
                config["logging"], DEFAULT_CONFIG["logging"]
            )
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = format_logging_config(
                config["logging"], DEFAULT_CONFIG["logging"]
            )
