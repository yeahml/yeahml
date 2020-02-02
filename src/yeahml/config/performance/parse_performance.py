from yeahml.config.default.util import parse_default


def format_performance_config(raw_config, DEFAULT):

    formatted_dict = {}
    formatted_dict = parse_default(raw_config, DEFAULT)

    return formatted_dict
