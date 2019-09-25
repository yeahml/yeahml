import os

import numpy as np

from yeahml.config.helper import create_standard_dirs


def format_meta_config(raw_config):
    formatted_dict = {}
    formatted_dict["name"] = raw_config["name"]
    formatted_dict["experiment_dir"] = raw_config["experiment_dir"]
    try:
        formatted_dict["seed"] = raw_config["rand_seed"]
    except KeyError:
        pass

    try:
        formatted_dict["trace_level"] = raw_config["trace"].lower()
    except KeyError:
        pass

    ### architecture
    # TODO: implement after graph can be created...
    formatted_dict["save_params"] = raw_config["saver"]["save_params_name"]

    try:
        formatted_dict["load_params_path"] = raw_config["saver"]["load_params_path"]
    except KeyError:
        # no params will be loaded from previously trained params
        pass

    # DEV_DIR is a hardcoded value for the directory in which the examples
    # are located. for packing+, this will need to be removed.
    DEV_DIR = "examples"
    # BEST_PARAMS_DIR is a hardcoded value that must match the created dir in
    # create_standard_dirs
    BEST_PARAMS_DIR = "best_params"
    formatted_dict["log_dir"] = os.path.join(
        ".", DEV_DIR, raw_config["name"], formatted_dict["experiment_dir"]
    )
    formatted_dict["save_weights_path"] = os.path.join(
        ".",
        DEV_DIR,
        raw_config["name"],
        formatted_dict["experiment_dir"],
        BEST_PARAMS_DIR,
        formatted_dict["save_params"] + ".h5",  # TODO: modify
    )
    # wipe is set to true for now
    create_standard_dirs(formatted_dict["log_dir"], True)

    return formatted_dict
