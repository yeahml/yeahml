import os
import pathlib
import shutil
from pathlib import Path

import tensorflow as tf

from yeahml.build.layers.config import return_available_layers
from yeahml.config.helper import create_standard_dirs
from yeahml.config.model.config import DEFAULT_ACT, IGNORE_HASH_KEYS
from yeahml.config.model.util import make_hash
from yeahml.config.default.util import parse_default

# from inspect import getmembers, isfunction


def _get_hidden_layers(raw_config: dict):
    try:
        hidden_layers = raw_config["layers"]
    except KeyError:
        raise KeyError("No Layers Found in the config")

    return hidden_layers


def _get_name_mapping(hl_type):
    # TODO: reverse many to one dict
    # NOTE: I'm not sure if/why this fn is needed - 20Oct19
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

    try:
        cur_func_vars = list(vars(func)["__init__"].__code__.co_varnames)
    except KeyError:
        # some layers inherit "__init__" from a base class e.g. batchnorm
        # the assumption here is that the 1st base class will contain the init..
        # I doubt this is accurate, but it is currently working
        try:
            first_parent = func.__bases__[0]
            cur_func_vars = list(vars(first_parent)["__init__"].__code__.co_varnames)
        except KeyError:
            raise NotImplementedError(
                f"error with type:{hl_type}, func:{func}, first parent: {first_parent}, other parents ({func.__bases__}). This error may be a result of an assumption that the __init__ params are from {first_parent} and not one of ({func.__bases__})"
            )
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

    # TODO: at some point, there should be a check to ensure this name is valid
    try:
        input_str = hl["input"]
    except KeyError:
        input_str = None
    HLD["input_str"] = input_str

    return HLD


def create_layer_config(hl: dict, default_activation: str) -> dict:
    HLD = {}

    ## TODO: parse layer type information
    HLD = parse_layer_type_information(hl, default_activation)

    return HLD


def format_model_config(raw_config: dict, DEFAULT: dict) -> dict:

    formatted_dict = {}
    formatted_dict = parse_default(raw_config, DEFAULT)

    # try:
    #     model_name = raw_config["meta"]["name"]
    # except KeyError:
    #     raise KeyError(
    #         "model:meta:name is not specified. Please specify a `name` in the model configuration"
    #     )
    # # model_root_dir = os.path.join(meta_dict["log_dir"], model_name)
    # model_root_dir = (
    #     Path(meta_dict["yeahml_dir"])
    #     .joinpath(meta_dict["data_name"])
    #     .joinpath(meta_dict["experiment_name"])
    # )
    # if pathlib.Path(model_root_dir).exists():
    #     try:
    #         override = raw_config["meta"]["name_override"]
    #     except KeyError:
    #         # default is False
    #         override = False
    #     if override:
    #         # wipe the directory to start fresh
    #         shutil.rmtree(model_root_dir)
    #     else:
    #         # TODO: logging
    #         raise ValueError(
    #             f"A model currently exists with the name {model_name}. If you wish to override the current model, you can use model:meta:name_override: True"
    #         )
    # else:
    #     pathlib.Path(model_root_dir).mkdir(parents=True, exist_ok=True)

    # try:
    #     default_activation = raw_config["meta"]["activation"]
    # except KeyError:
    #     default_activation = DEFAULT_ACT
    # # create architecture config
    # formatted_config = {}
    # formatted_config["model_root_dir"] = model_root_dir
    # hidden_layers = _get_hidden_layers(raw_config)
    # if not hidden_layers:
    #     raise ValueError(
    #         f"hidden layer field was found, but did not contain any layers: {raw_config}"
    #     )

    # approved_layers_config = {}
    # for hl in hidden_layers:
    #     approved_layers_config[hl] = create_layer_config(
    #         hidden_layers[hl], default_activation
    #     )

    # formatted_config["layers"] = approved_layers_config

    # # model specific directories

    # # TODO: I really don't like this save/<xxx> naming...
    # new_dirs = create_standard_dirs(
    #     model_root_dir, ["save/model", "save/params", "tf_logs", "yf_logs"], False
    # )
    # # formatted_dict["save_weights_dir"] = os.path.join(
    # #     model_root_dir, save_dir, "params"
    # # )
    # # formatted_dict["save_model_dir"] = os.path.join(model_root_dir, save_dir, "model")

    # formatted_config = {**formatted_config, **new_dirs}

    # # add a model hash
    # # TODO: eventually, this could be used to track model architectures
    # model_hash = make_hash(formatted_config, IGNORE_HASH_KEYS)
    # formatted_config["model_hash"] = model_hash

    return formatted_dict
