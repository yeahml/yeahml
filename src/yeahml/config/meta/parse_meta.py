from pathlib import Path

from yeahml.config.helper import create_exp_dir


def format_meta_config(raw_config, DEFAULT):
    formatted_dict = {}
    # formatted_dict["name"] = raw_config["name"]
    # formatted_dict["experiment_dir"] = raw_config["experiment_name"]
    # try:
    #     formatted_dict["seed"] = raw_config["rand_seed"]
    # except KeyError:
    #     pass

    # try:
    #     formatted_dict["trace_level"] = raw_config["trace"].lower()
    # except KeyError:
    #     pass

    # ### architecture
    # try:
    #     formatted_dict["load_params_path"] = raw_config["saver"]["load_params_path"]
    # except KeyError:
    #     # no params will be loaded from previously trained params
    #     pass

    # # directory for which yeahml should save information
    # try:
    #     yml_dir = raw_config["yeahml_dir"]
    # except KeyError:
    #     # no params will be loaded from previously trained params
    #     yml_dir = "yeahml"

    for k, k_config in DEFAULT.items():
        try:
            cur = raw_config[k]
        except KeyError:
            if k_config["required"]:
                raise KeyError(f"{k} is required, but is not specified")
            else:
                cur = k_config["default"]
        formatted_dict[k] = cur

    # formatted_dict["log_dir"] = (
    #     Path(formatted_dict["yeahml"])
    #     .joinpath(formatted_dict["name"])
    #     .joinpath(formatted_dict["experiment_name"])
    # )

    # TODO: this likely needs to be pushed to where it is implemented
    create_exp_dir(
        Path(formatted_dict["yeahml_dir"])
        .joinpath(formatted_dict["data_name"])
        .joinpath(formatted_dict["experiment_name"]),
        wipe_dirs=False,
    )

    return formatted_dict
