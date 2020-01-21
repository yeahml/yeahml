from yeahml.config.helper import get_raw_dict_from_string
from yeahml.config.model.parse_model import format_model_config


def config_model(model_string: str, config: dict):
    model_config_raw = get_raw_dict_from_string(model_string)

    formatted_model_config = format_model_config(model_config_raw, config["meta"])

    # keep previous config constant, but update model architecture
    new_config = config.copy()
    new_config["model"] = formatted_model_config

    return new_config
