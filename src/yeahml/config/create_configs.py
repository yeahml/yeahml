#!/usr/bin/env python
import functools
import operator
import os
import shutil
from pathlib import Path
from typing import List
import crummycm as ccm

from yeahml.config.default.types.compound.layer import layers_parser
from yeahml.config.default.types.compound.performance import performances_parser

from yeahml.config.template.template import TEMPLATE

from yeahml.config.default.default_config import DEFAULT_CONFIG
from yeahml.config.default.util import parse_default
from yeahml.config.graph_analysis.static_analysis import static_analysis

# from yeahml.config.graph_analysis import static_analysis
from yeahml.config.helper import extract_dict_from_path, get_raw_dict_from_string
from yeahml.config.model.config import IGNORE_HASH_KEYS
from yeahml.config.model.util import make_hash
from yeahml.log.yf_logging import config_logger

## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used


def _maybe_create_dir(root_dir: str, wipe_dirs: bool, logger=None):

    if os.path.exists(root_dir):
        if wipe_dirs:
            shutil.rmtree(root_dir)
            logger.info(f"directory {root_dir} removed")
        else:
            raise ValueError(
                f"a model experiment directory currently exists at {root_dir}."
                " If you wish to override the current model, you can use"
                " meta:start_fresh: True"
            )

    if not os.path.exists(root_dir):
        Path(root_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"directory {root_dir} created")


def create_configs(main_path: str) -> dict:

    # parse + validate
    config_dict = ccm.generate(main_path, TEMPLATE)

    # TODO: bandaid fix
    if "callbacks" not in config_dict.keys():
        config_dict["callbacks"] = None

    config_dict["model"]["layers"] = layers_parser()(config_dict["model"]["layers"])

    config_dict["performance"]["objectives"] = performances_parser()(
        config_dict["performance"]["objectives"]
    )

    # TODO: ---- below
    model_hash = make_hash(config_dict["model"], IGNORE_HASH_KEYS)
    config_dict["model"]["model_hash"] = model_hash

    full_exp_path = (
        Path(config_dict["meta"]["yeahml_dir"])
        .joinpath(config_dict["meta"]["data_name"])
        .joinpath(config_dict["meta"]["experiment_name"])
        .joinpath(config_dict["model"]["name"])
    )
    logger = config_logger(full_exp_path, config_dict["logging"], "config")

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
            logger.info(f"directory {exp_root_dir} removed")

    if not os.path.exists(exp_root_dir):
        Path(exp_root_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"directory {exp_root_dir} created")

    model_root_dir = exp_root_dir.joinpath(config_dict["model"]["name"])
    try:
        override_model_dir = config_dict["model"]["start_fresh"]
    except KeyError:
        # leave existing model information
        override_model_dir = False

    _maybe_create_dir(model_root_dir, wipe_dirs=override_model_dir, logger=logger)

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
    graph_dict, graph_dependencies = static_analysis(config_dict)
    config_dict["graph_dict"] = graph_dict
    config_dict["graph_dependencies"] = graph_dependencies

    return config_dict
