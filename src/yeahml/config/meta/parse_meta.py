from pathlib import Path

from yeahml.config.helper import create_exp_dir


def format_meta_config(raw_config):
    formatted_dict = {}
    formatted_dict["name"] = raw_config["name"]
    formatted_dict["experiment_dir"] = raw_config["experiment_name"]
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

    # directory for which yeahml should save information
    try:
        yml_dir = raw_config["yeahml_dir"]
    except KeyError:
        # no params will be loaded from previously trained params
        yml_dir = "yeahml"

    formatted_dict["log_dir"] = (
        Path(yml_dir)
        .joinpath(raw_config["name"])
        .joinpath(formatted_dict["experiment_dir"])
    )
    # wipe is set to True for now
    create_exp_dir(formatted_dict["log_dir"], wipe_dirs=False)

    return formatted_dict
