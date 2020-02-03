from yeahml.config.default.util import parse_default


def format_logging_config(raw_config, DEFAULT):
    formatted_dict = parse_default(raw_config, DEFAULT)
    return formatted_dict
