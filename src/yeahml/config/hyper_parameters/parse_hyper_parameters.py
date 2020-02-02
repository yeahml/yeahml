from yeahml.config.default.util import parse_default


def format_hyper_parameters_config(raw_config: dict, DEFAULT: dict) -> dict:
    formatted_config = {}
    formatted_config = parse_default(raw_config, DEFAULT)
    return formatted_config
