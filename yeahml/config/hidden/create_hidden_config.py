from inspect import getmembers, isfunction

import tensorflow as tf

from yeahml.build.layers import config
from yeahml.build.layers.config import return_available_layers
from yeahml.config.helper import parse_yaml_from_path
from yeahml.config.hidden.config import DEFAULT_ACT


def _get_hidden_layers(h_raw_config: dict):
    try:
        hidden_layers = h_raw_config["layers"]
    except KeyError:
        raise KeyError("No Layers Found in the config")

    return hidden_layers


def _get_name_mapping(hl_type):
    # TODO: reverse many to one dict
    # naming_dict = {}
    return hl_type


def parse_layer_type_information(hl: dict, default_activation: str) -> dict:
    # parse layer specific information and ensure the options are
    # "reasonable" for each layer
    # set "type", "options"
    HLD = {}

    LAYER_FUNCTIONS = return_available_layers()

    try:
        hl_type = hl["type"].lower()
        # name correction
        hl_type = _get_name_mapping(hl_type)
        HLD["type"] = hl_type
    except KeyError:
        raise ValueError(f"layer does not have a 'type': {hl}")

    # TODO: could place a hl_type = get_hl_type_mapping() here
    if hl_type in LAYER_FUNCTIONS.keys():
        func = LAYER_FUNCTIONS[hl_type]["function"]
    else:
        raise NotImplementedError(
            f"layer type {hl_type} not implemented yet. Current supported layers are: {LAYER_FUNCTIONS.keys()}"
        )

    cur_func_vars = list(vars(func)["__init__"].__code__.co_varnames)
    if issubclass(func, tf.keras.layers.Layer):
        # replace "kwargs"
        cur_func_vars.remove("kwargs")
        cur_func_vars.extend(["trainable", "name", "dtype", "dynamic"])
        # TODO: fix this so it isn't hard coded to layer
        # > print(vars(tf.keras.layers.Layer)["__init__"].__code__.co_varnames)
        # > print(tf.keras.layers.Layer.__code__.co_varnames)

    # ensure all options are allowed by the specific function
    try:
        opts_raw = hl["options"]
    except:
        opts_raw = None  # default to default options (tf)
    if opts_raw:
        for opt in opts_raw:
            if opt not in cur_func_vars:
                raise ValueError(f"option {opt} is not allowed by type {func}")
    HLD["options"] = opts_raw  # TODO: opts_formatted?

    # Not all layers require an activation function
    if "activation" in cur_func_vars:
        try:
            actfn_str = hl["options"]["activation"]
        except KeyError:
            try:
                # fall back to the default activation function specified
                actfn_str = default_activation
            except KeyError:
                actfn_str = DEFAULT_ACT
        HLD["options"]["activation"] = actfn_str

    return HLD


def create_layer_config(hl: dict, default_activation: str) -> dict:
    HLD = {}

    ## TODO: parse layer type information
    HLD = parse_layer_type_information(hl, default_activation)

    return HLD


def extract_hidden_dict_and_set_defaults(
    h_raw_config: dict, default_activation: str
) -> dict:
    # create architecture config
    parsed_h_config = {}
    hidden_layers = _get_hidden_layers(h_raw_config)
    if not hidden_layers:
        raise ValueError(
            f"hidden layer field was found, but did not contain any layers: {h_raw_config}"
        )

    approved_layers_config = {}
    for hl in hidden_layers:
        approved_layers_config[hl] = create_layer_config(
            hidden_layers[hl], default_activation
        )

    parsed_h_config["layers"] = approved_layers_config

    return parsed_h_config
