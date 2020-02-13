from yeahml.config.default.util import parse_default


def format_data_config(raw_config: dict, DEFAULT: dict):
    formatted_dict = {}
    formatted_dict = parse_default(raw_config, DEFAULT)
    return formatted_dict
