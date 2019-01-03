#!/usr/bin/env python
import logging
import sys

from yeahml.config.helper import maybe_create_dir, parse_yaml_from_path
from yeahml.config.model.create_model_config import extract_model_dict_and_set_defaults
from yeahml.config.hidden.create_hidden_config import (
    extract_hidden_dict_and_set_defaults,
)


## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used


def create_model_and_hidden_config(path: str) -> tuple:
    # return the model and architecture configuration dicts
    raw_config = parse_yaml_from_path(path)
    if not raw_config:
        sys.exit(
            "Error > Exiting: the model config file was found {}, but appears to be empty".format(
                path
            )
        )

    m_config = extract_model_dict_and_set_defaults(raw_config)
    h_config = extract_hidden_dict_and_set_defaults(raw_config)

    return (m_config, h_config)
