from pathlib import Path

from yeahml.config.helper import create_exp_dir
from yeahml.config.default.util import parse_default


def format_meta_config(raw_config, DEFAULT):

    formatted_dict = parse_default(raw_config, DEFAULT)

    return formatted_dict
