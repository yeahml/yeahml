from pathlib import Path

from yeahml.config.helper import create_exp_dir
from yeahml.config.default.util import parse_default


def format_meta_config(raw_config, DEFAULT):

    formatted_dict = parse_default(raw_config, DEFAULT)

    # TODO: this likely needs to be pushed to where it is implemented
    create_exp_dir(
        Path(formatted_dict["yeahml_dir"])
        .joinpath(formatted_dict["data_name"])
        .joinpath(formatted_dict["experiment_name"]),
        wipe_dirs=False,
    )

    return formatted_dict
