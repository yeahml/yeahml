import os

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
    save_dir = "save"
    formatted_dict["log_dir"] = os.path.join(
        ".", DEV_DIR, raw_config["name"], formatted_dict["experiment_dir"]
    )
    formatted_dict["save_weights_path"] = os.path.join(
        ".",
        DEV_DIR,
        raw_config["name"],
        formatted_dict["experiment_dir"],
        save_dir,
        "params",
        "best_params" + ".h5",  # TODO: modify
    )
    formatted_dict["save_model_path"] = os.path.join(
        ".",
        DEV_DIR,
        raw_config["name"],
        formatted_dict["experiment_dir"],
        save_dir,
        "model",
        "model.h5",
    )

    # wipe is set to true for now
    new_dirs = create_standard_dirs(
        formatted_dict["log_dir"],
        ["save/model", "save/params", "tf_logs", "yf_logs"],
        True,
    )
    formatted_dict = {**formatted_dict, **new_dirs}

    return formatted_dict
