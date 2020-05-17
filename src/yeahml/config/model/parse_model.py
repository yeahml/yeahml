from yeahml.config.default.util import parse_default
from yeahml.config.model.config import IGNORE_HASH_KEYS
from yeahml.config.model.util import make_hash


def format_model_config(raw_config: dict, DEFAULT: dict) -> dict:

    formatted_dict = {}
    formatted_dict = parse_default(raw_config, DEFAULT)

    # # add a model hash
    # # TODO: eventually, this could be used to track model architectures
    model_hash = make_hash(formatted_dict, IGNORE_HASH_KEYS)
    formatted_dict["model_hash"] = model_hash

    return formatted_dict
