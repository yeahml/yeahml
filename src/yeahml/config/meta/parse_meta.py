import os

from yeahml.config.helper import create_exp_dir


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
    try:
        formatted_dict["load_params_path"] = raw_config["saver"]["load_params_path"]
    except KeyError:
        # no params will be loaded from previously trained params
        pass

    # DEV_DIR is a hardcoded value for the directory in which the examples
    # are located. for packing+, this will need to be removed.
    DEV_DIR = "examples"
    # save_dir is a hardcoded value that must match the created dir in
    # create_standard_dirs
    formatted_dict["log_dir"] = os.path.join(
        ".", DEV_DIR, raw_config["name"], formatted_dict["experiment_dir"]
    )
    # wipe is set to True for now
    create_exp_dir(formatted_dict["log_dir"], wipe_dirs=False)

    return formatted_dict
