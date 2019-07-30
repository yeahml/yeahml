import sys

from yeahml.config.helper import parse_yaml_from_path
from yeahml.config.hidden.components.pooling import configure_pooling_layer

from yeahml.build.layers.config import LAYER_FUNCTIONS
from inspect import getmembers, isfunction
from yeahml.build.layers import config

import tensorflow as tf
import inspect

# from yeahml.config.hidden.components.convolution import configure_conv_layer


def _get_hidden_layers(h_raw_config: dict):
    try:
        hidden_layers = h_raw_config["layers"]
    except KeyError:
        sys.exit("No Layers Found")

    return hidden_layers


def parse_layer_type_information(hl: dict, default_activation: str) -> dict:
    # parse layer specific information and ensure the options are
    # "reasonable" for each layer
    # set "type", "options", and "activation"
    HLD = {}

    try:
        hl_type = hl["type"].lower()
        HLD["type"] = hl_type
    except KeyError:
        sys.exit("layer does not have a 'type': {}".format(hl))

    # TODO: could place a hl_type = get_hl_type_mapping() here
    if hl_type in LAYER_FUNCTIONS.keys():
        func = LAYER_FUNCTIONS[hl_type]["function"]
    else:
        raise NotImplementedError(f"layer type {hl_type} not implemented yet")

    print(func)
    allvars = list(vars(func)["__init__"].__code__.co_varnames)
    # If of type layer, add kwargs
    if issubclass(func, tf.keras.layers.Layer):
        allvars.remove("kwargs")
        # TODO: fix this so it isn't hard coded to layer
        # print(vars(tf.keras.layers.Layer)["__init__"].__code__.co_varnames)
        # print(tf.keras.layers.Layer.__code__.co_varnames)
        # print(vars(tf.keras.layers.Layer)["__init__"].__code__.co_varnames)
        allvars.extend(["trainable", "name", "dtype", "dynamic"])

    # get name of current function
    # get name of all functions defined within a module
    # func_name = func.__name__
    # print(func_name)
    # avail_functions = [o for o in getmembers(config) if isfunction(o[1])]
    # print(avail_functions)
    # fidx = [name for name, fnptr in enumerate(avail_functions) if fnptr[0] == func_name]
    # if len(fidx) == 1:
    #     fidx = fidx[0]
    # else:
    #     raise ValueError(f"No function available for {func_name}")

    # ('build_conv_layer', <function build_conv_layer at 0x7f34d6f09048>)
    # [1] is used to get the actual function
    # fn_kwargs = avail_functions[fidx][1].__code__.co_varnames
    # print(getmembers(avail_functions[fidx][1]))
    # print(avail_functions[fidx][1].__code__)

    # print(fn_kwargs)
    # print(hl["options"])
    # aa = getmembers(avail_functions[fidx][1])
    # print(aa.base_function)
    # print(aa["__init__"])
    # iindx = [i for i, (n, f) in enumerate(aa) if n == "__code__"][0]
    # print(aa[iindx][1].co_varnames)
    # for a in aa:
    #     print(a)

    try:
        opts_raw = hl["options"]
    except:
        # will default to default options
        opts_raw = None

    if opts_raw:
        for opt in opts_raw:
            if opt in allvars:
                pass
            else:
                raise ValueError(f"option {opt} is not allowed by type {func}")

    # print(getattr(avail_functions[fidx][1], "base_function"))
    # all_vars = vars(base_function)["__init__"].__code__.co_varnames
    # sys.exit("done")
    ## option logic for each layer type
    # see if options exist

    # opts_formatted = {}
    # if hl_type == "conv2d":
    #     opts_formatted = configure_conv_layer(opts_raw)
    # elif hl_type == "deconv2d":
    #     opts_formatted = configure_conv_layer(opts_raw)
    # elif hl_type == "deconv2d":
    #     pass
    # elif hl_type == "dense":
    #     pass
    # elif hl_type == "deconv2d":
    #     pass
    # elif hl_type == "pooling2d":
    #     opts_formatted = configure_pooling_layer(opts_raw)
    # elif hl_type == "pooling1d":
    #     opts_formatted = configure_pooling_layer(opts_raw)
    # elif hl_type == "global_pooling":
    #     pass
    # elif hl_type == "embedding":
    #     pass
    # elif hl_type == "batch_normalization":
    #     pass
    # elif hl_type == "recurrent":
    #     pass
    # else:
    #     sys.exit("layer type {} not currently supported".format(hl_type))
    HLD["options"] = opts_raw  # opts_formatted

    # TODO: this needs to be pushed down a level since some layers doesn't require act
    # if "actfn" in func.__code__.co_varnames:
    try:
        actfn_str = hl["activation"]
    except KeyError:
        try:
            # fall back to the default activation function specified
            actfn_str = default_activation
        except KeyError:
            # relu: the reasoning here is that the relu is subjectively the most
            # common/default activation function in DNNs, but I don't LOVE this
            actfn_str = "relu"
    HLD["activation"] = actfn_str
    # TODO: logger.debug("activation set: {}".format(actfn_str))

    return HLD


def create_layer_config(hl: dict, default_activation: str) -> dict:
    HLD = {}

    ## TODO: parse layer type information
    HLD = parse_layer_type_information(hl, default_activation)

    return HLD


def extract_hidden_dict_and_set_defaults(
    h_raw_config: dict, default_activation: str
) -> dict:
    parsed_h_config = {}
    # create architecture config

    hidden_layers = _get_hidden_layers(h_raw_config)
    if not hidden_layers:
        sys.exit("a hidden layer field was found, but did not contain any layers")

    approved_layers_config = {}
    for hl in hidden_layers:
        approved_layers_config[hl] = create_layer_config(
            hidden_layers[hl], default_activation
        )

    parsed_h_config["layers"] = approved_layers_config

    return parsed_h_config
