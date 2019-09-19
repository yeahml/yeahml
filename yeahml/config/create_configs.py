#!/usr/bin/env python
import logging
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


def create_configs(path: str) -> tuple:
    # return the model and architecture configuration dicts
    if path.endswith("yaml") or path.endswith("yml"):
        main_config_raw = parse_yaml_from_path(path)
    elif path.endswith("json"):
        main_config_raw = parse_json_from_path(path)
    if not main_config_raw:
        raise ValueError(
            f"Error > Exiting: the model config file was found {path}, but appears to be empty"
        )

    main_cdict = extract_model_dict_and_set_defaults(main_config_raw)

    # TODO: does this belong here?
    try:
        def_act = main_cdict["def_act"]
    except KeyError:
        def_act = None

    # get model
    try:
        model_path = main_config_raw["model"]["path"]
    except KeyError:
        raise KeyError("no path specified for the hidden layers (:'hidden':'path')")

    if model_path.endswith("yaml") or model_path.endswith("yml"):
        model_config_raw = parse_yaml_from_path(model_path)
    elif model_path.endswith("json"):
        model_config_raw = parse_json_from_path(model_path)
    if not model_config_raw:
        raise ValueError(
            f"Error > Exiting: the model config file was found {path}, but appears to be empty"
        )

    model_cdict = extract_hidden_dict_and_set_defaults(model_config_raw, def_act)

    return (main_cdict, model_cdict)
