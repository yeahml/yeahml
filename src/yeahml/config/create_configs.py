#!/usr/bin/env python
import functools
import operator
import os
import shutil
from pathlib import Path
from typing import List

from yeahml.config.default.default_config import DEFAULT_CONFIG
from yeahml.config.default.util import parse_default

# from yeahml.config.graph_analysis import static_analysis
from yeahml.config.helper import extract_dict_from_path, get_raw_dict_from_string
from yeahml.config.model.config import IGNORE_HASH_KEYS
from yeahml.config.model.util import make_hash
from yeahml.log.yf_logging import config_logger

from yeahml.config.graph_analysis.static_analysis import static_analysis_v2
from yeahml.config.model.util import make_hash

## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used

# TODO: I don't like this global, but I'm not sure where it belongs yet
# NOTE: it is required that meta be created before model. this may need to change
CONFIG_KEYS = [
    "meta",
    "logging",
    "performance",
    "data",
    "hyper_parameters",
    "model",
    "optimize",
]


def _maybe_extract_from_path(cur_dict: dict) -> dict:
    try:
        cur_path = cur_dict["path"]
        if len(cur_dict.keys()) > 1:
            raise ValueError(
                "the current dict has a path specified, but also contains other"
                f" top level keys ({cur_dict.keys()}). please move these keys"
                " to path location or remove"
            )
        cur_dict = extract_dict_from_path(cur_path)
    except KeyError:
        pass
    return cur_dict


def _create_exp_dir(root_dir: str, wipe_dirs: bool):

    if os.path.exists(root_dir):
        if wipe_dirs:
            shutil.rmtree(root_dir)
        else:
            raise ValueError(
                f"a model experiment directory currently exists at {root_dir}."
                " If you wish to override the current model, you can use"
                " meta:start_fresh: True"
            )

    if not os.path.exists(root_dir):
        Path(root_dir).mkdir(parents=True, exist_ok=True)


def check_for_unused_keys(
    conf_formatted: dict,
    conf_raw: dict,
    parent_key_path: List[str],
    unused_keys: List[List[str]],
) -> List[List[str]]:
    """Recursive function to check for unparsed keys in the user's config.

    The idea is to have the user's config only contain information that is known
    to yeahml. This will help prevent confusion about what is/isn't being used
    by the framework.

    TODO: if a `path` is specified (as will likely be typical for the model
    def), the extra key check is currently ignored - 13June20

    Parameters
    ----------
    conf_formatted : dict
        The yeahml formatted+parsed config
    conf_raw : dict
        The user's raw config
    parent_key_path : List[str]
        Key path that has lead to the current location
    unused_keys : List[List[str]]
        Nested keys to unparsed values

    Returns
    -------
    List[List[str]]
        Nested keys to unparsed values
    """
    raw_keys = conf_raw.keys()
    for key in raw_keys:
        cur_key_path = parent_key_path.copy()
        cur_key_path.append(key)

        if key == "path":
            # TODO: "path" is a magic word indicating that in the users config,
            # the nested structure for the indicated section is written
            # somewhere else (path)
            # TODO: this isn't perfectly straight forward, i.e. the config for a
            # model is not 1:1 with the returned formatted config
            # raw = _maybe_extract_from_path(conf_raw)
            # check_for_unused_keys(conf_formatted, raw, cur_key_path,
            # unused_keys)
            pass
        elif key == "instructions" and cur_key_path[-2] == "directive":
            # TODO: this is a bandaid fix -- the issue is that the return type
            # of [optimize][directive][instructions] is a dict and thus is
            # attempted to be checked against below by check_for_unused_keys and
            # so the solution currently is to not check this key
            return
        else:
            raw = conf_raw[key]

        if key not in conf_formatted.keys():
            unused_keys.append(cur_key_path)
        else:
            obj = conf_formatted[key]
            if isinstance(obj, dict):
                check_for_unused_keys(obj, raw, cur_key_path, unused_keys)

    return unused_keys


def _getFromDict(dataDict, mapList):
    # https://stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys
    return functools.reduce(operator.getitem, mapList, dataDict)


def _build_unused_keys_message(main_config_raw, unused_keys):
    unchecked_str = ""
    error_str = ""
    for uks in unused_keys:
        retv = _getFromDict(main_config_raw, uks)
        if uks[-1] == "path":
            unchecked_str += f" - main_config [{']['.join(uks)}] = {retv}\n"
        else:
            error_str += f" - main_config [{']['.join(uks)}] = {retv}\n"
    return error_str, unchecked_str


def _maybe_message(unused_keys, main_config_raw, logger):
    err_str, unchecked_str = _build_unused_keys_message(main_config_raw, unused_keys)
    if unchecked_str:
        unchecked_msg = (
            f"The following paths were not checked for unparsed keys:\n"
            f"{unchecked_str}"
        )
    else:
        unchecked_msg = ""
    if err_str:
        raise ValueError(
            (
                "Unparsed keys detected\n\n"
                f"{unchecked_msg}"
                f"The following unparsed keys were detected:\n"
                f"{err_str}"
                "please remove these keys from the config.\n\n"
                "If you would like to support these keys; \n"
                "please modify the `DEFAULT_CONFIG` in `src/yeahml/config/default/default_config.py`"
            )
        )
    elif unchecked_str:
        logger.warn(
            "some keys have not been checked for unparsed keys:\n"
            f"{unchecked_str}"
            "\n This is common when the config contains model:path: <some_path>.yml and means that"
            " there could be extra keys in this model config that are not being parsed+used by yeahml."
            " in a future release, this could be addressed by building parsing logic in `src/yeahml/config/create_configs.py`"
        )


def _primary_config(main_path: str) -> dict:
    main_config_raw = get_raw_dict_from_string(main_path)
    cur_keys = main_config_raw.keys()
    invalid_keys = []
    for key in CONFIG_KEYS:
        if key not in cur_keys:
            invalid_keys.append(key)
            # not all of these *need* to be present, but for now that will be
            # enforced
    if invalid_keys:
        raise ValueError(
            f"The main config does not contain the key(s) {invalid_keys}:"
            f" current keys: {cur_keys}"
        )

    # build dict containing configs
    config_dict = {}
    for config_type in CONFIG_KEYS:
        # try block?
        raw_config = main_config_raw[config_type]
        raw_config = _maybe_extract_from_path(raw_config)

        formatted_config = parse_default(raw_config, DEFAULT_CONFIG[f"{config_type}"])
        if config_type == "model":
            model_hash = make_hash(formatted_config, IGNORE_HASH_KEYS)
            formatted_config["model_hash"] = model_hash

        config_dict[config_type] = formatted_config

    full_exp_path = (
        Path(config_dict["meta"]["yeahml_dir"])
        .joinpath(config_dict["meta"]["data_name"])
        .joinpath(config_dict["meta"]["experiment_name"])
        .joinpath(config_dict["model"]["name"])
    )
    logger = config_logger(full_exp_path, config_dict["logging"], "config")

    unused_keys = check_for_unused_keys(config_dict, main_config_raw, [], [])
    if unused_keys:
        _maybe_message(unused_keys, main_config_raw, logger)

    # TODO: this should probably be made once and stored? in the :meta?
    exp_root_dir = (
        Path(config_dict["meta"]["yeahml_dir"])
        .joinpath(config_dict["meta"]["data_name"])
        .joinpath(config_dict["meta"]["experiment_name"])
    )

    try:
        override_yml_dir = config_dict["meta"]["start_fresh"]
    except KeyError:
        # leave existing model information
        override_yml_dir = False

    if os.path.exists(exp_root_dir):
        if override_yml_dir:
            shutil.rmtree(exp_root_dir)
    if not os.path.exists(exp_root_dir):
        Path(exp_root_dir).mkdir(parents=True, exist_ok=True)

    model_root_dir = exp_root_dir.joinpath(config_dict["model"]["name"])
    try:
        override_model_dir = config_dict["model"]["start_fresh"]
    except KeyError:
        # leave existing model information
        override_model_dir = False

    _create_exp_dir(model_root_dir, wipe_dirs=override_model_dir)

    return config_dict


def create_configs(main_path: str) -> dict:

    # parse individual configs
    config_dict = _primary_config(main_path)

    # build the order of inputs into the model. This logic will likely need to
    # change as inputs become more complex
    input_order = []
    for ds_name, ds_config in config_dict["data"]["datasets"].items():
        for feat_name, config in ds_config["in"].items():
            if config["startpoint"]:
                if not config["label"]:
                    input_order.append(feat_name)
    if not input_order:
        raise ValueError("no inputs have been specified to the model")

    # loop model to ensure all outputs are accounted for
    output_order = []
    for name, config in config_dict["model"]["layers"].items():
        if config["endpoint"]:
            output_order.append(name)
    if not output_order:
        raise ValueError("no outputs have been specified for the model")

    # TODO: maybe this should be a dictionary
    # TODO: this is a sneaky way + band-aid of ensuring we don't specify inputs
    # if they are named the same -- in reality this does not address the root
    # issue, that is that we should be able to allow some intermediate layers to
    # accept input from either layer_a or layer_b, not only layer_a
    input_order = list(set(input_order))

    config_dict["model_io"] = {"inputs": input_order, "outputs": output_order}

    # validate graph
    # static_dict, subgraphs = static_analysis(config_dict)
    # graph_dict, subgraphs
    graph_dict, graph_dependencies = static_analysis_v2(config_dict)

    config_dict["graph_dict"] = graph_dict
    config_dict["graph_dependencies"] = graph_dependencies

    # config_dict["static"] = static_dict
    # config_dict["subgraphs"] = subgraphs

    return config_dict
