#!/usr/bin/env python
import logging
import sys

from yeahml.config.helper import (
    maybe_create_dir,
    parse_yaml_from_path,
    parse_json_from_path,
)
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
    if path.endswith("yaml") or path.endswith("yml"):
        raw_config = parse_yaml_from_path(path)
    elif path.endswith("json"):
        raw_config = parse_json_from_path(path)
    if not raw_config:
        sys.exit(
            "Error > Exiting: the model config file was found {}, but appears to be empty".format(
                path
            )
        )

    m_config = extract_model_dict_and_set_defaults(raw_config)

    # get hidden path
    try:
        hidden_path = raw_config["hidden"]["path"]
    except KeyError:
        sys.exit("no path specified for the hidden layers")

    if hidden_path.endswith("yaml") or hidden_path.endswith("yml"):
        h_raw_config = parse_yaml_from_path(hidden_path)
    elif hidden_path.endswith("json"):
        h_raw_config = parse_json_from_path(hidden_path)
    if not h_raw_config:
        sys.exit(
            "Error > Exiting: the model config file was found {}, but appears to be empty".format(
                path
            )
        )

    # TODO: does this belong here?
    try:
        def_act = m_config["def_act"]
    except KeyError:
        def_act = None

    h_config = extract_hidden_dict_and_set_defaults(h_raw_config, def_act)

    return (m_config, h_config)
