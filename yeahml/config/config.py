from yeahml.config.helper import get_raw_dict_from_string
from yeahml.config.model.parse_model import format_model_config


def config_model(model_string: str, meta_dict: dict):
    model_config_raw = get_raw_dict_from_string(model_string)
    formatted_config = format_model_config(model_config_raw, meta_dict)
    return formatted_config
