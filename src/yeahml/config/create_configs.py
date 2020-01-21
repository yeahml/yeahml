#!/usr/bin/env python

from yeahml.config.data.parse_data import format_data_config
from yeahml.config.helper import extract_dict_from_path, get_raw_dict_from_string
from yeahml.config.hyper_parameters.parse_hyper_parameters import (
    format_hyper_parameters_config,
)
from yeahml.config.logging.parse_logging import format_logging_config
from yeahml.config.meta.parse_meta import format_meta_config
from yeahml.config.model.parse_model import format_model_config
from yeahml.config.performance.parse_performance import format_performance_config

# components

## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used

# TODO: I don't like this global, but I'm not sure where it belongs yet
# NOTE: it is required that meta be created before model. this may need to change
CONFIG_KEYS = ["meta", "logging", "performance", "data", "hyper_parameters", "model"]


def maybe_extract_from_path(cur_dict: dict) -> dict:
    try:
        cur_path = cur_dict["path"]
        if len(cur_dict.keys()) > 1:
            raise ValueError(
                f"the current dict has a path specified, but also contains other top level keys ({cur_dict.keys()}). please move these keys to path location or remove"
            )
        cur_dict = extract_dict_from_path(cur_path)
    except KeyError:
        pass
    return cur_dict


def create_configs(main_path: str) -> dict:
    main_config_raw = get_raw_dict_from_string(main_path)
    cur_keys = main_config_raw.keys()
    invalid_keys = []
    for key in CONFIG_KEYS:
        if key not in cur_keys:
            invalid_keys.append(key)
            # not all of these *need* to be present, but for now that will be enforced
    if invalid_keys:
        raise ValueError(
            f"The main config does not contain the key(s) {invalid_keys}: current keys: {cur_keys}"
        )

    # build dict containing configs
    config_dict = {}
    for config_type in CONFIG_KEYS:
        raw_config = main_config_raw[config_type]
        # if the main key has a "path" key, then extract from that path
        raw_config = maybe_extract_from_path(raw_config)
        if config_type == "meta":
            formatted_config = format_meta_config(raw_config)
        elif config_type == "logging":
            formatted_config = format_logging_config(raw_config)
        elif config_type == "performance":
            formatted_config = format_performance_config(raw_config)
        elif config_type == "data":
            formatted_config = format_data_config(raw_config)
        elif config_type == "hyper_parameters":
            formatted_config = format_hyper_parameters_config(raw_config)
        elif config_type == "model":
            formatted_config = format_model_config(raw_config, config_dict["meta"])
        else:
            raise ValueError(f"config type {config_type} is not yet implemented")
        config_dict[config_type] = formatted_config

    return config_dict
