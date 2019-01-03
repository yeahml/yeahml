#!/usr/bin/env python
import logging
import sys

from yeahml.config.helper import maybe_create_dir, parse_yaml_from_path
from yeahml.config.manage_parameters import extract_dict_and_set_defaults


## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used


def create_model_and_hidden_config(path: str) -> tuple:
    # return the model and architecture configuration dicts
    m_config = parse_yaml_from_path(path)
    if not m_config:
        sys.exit(
            "Error > Exiting: the model config file was found {}, but appears to be empty".format(
                path
            )
        )
    # create architecture config
    if m_config["hidden"]["yaml"]:
        h_config = parse_yaml_from_path(m_config["hidden"]["yaml"])
    else:
        # hidden is defined in the current yaml
        # TODO: this needs error checking/handling, empty case
        h_config = m_config["hidden"]

    m_config, h_config = extract_dict_and_set_defaults(m_config, h_config)

    return (m_config, h_config)
