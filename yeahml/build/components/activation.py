import inspect

import tensorflow as tf

from yeahml.build.components.util import copy_func


def _configure_activation(opt_dict):
    # TODO: this is dangerous.... (updating the __defaults__ like this)
    act_fn = return_activation(opt_dict["type"])["function"]
    act_fn = copy_func(act_fn)
    temp_copy = opt_dict.copy()
    _ = temp_copy.pop("type")
    if temp_copy:
        var_list = list(act_fn.__code__.co_varnames)
        cur_defaults_list = list(act_fn.__defaults__)
        # TODO: try?
        var_list.remove("x")
        for ao, v in temp_copy.items():
            arg_index = var_list.index(ao)
            # TODO: same type assertion?
            cur_defaults_list[arg_index] = v
        act_fn.__defaults__ = tuple(cur_defaults_list)

    return act_fn


def return_available_activations():
    # logic to get all layers in a class
    ACTIVATION_FUNCTIONS = {}
    available_keras_activations = tf.keras.activations.__dict__

    for opt_name, opt_func in available_keras_activations.items():
        if callable(opt_func):
            # TODO: likely a better way to handle this.. but I'm unsure how personally
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                ACTIVATION_FUNCTIONS[opt_name.lower()] = {}
                ACTIVATION_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = list(opt_func.__code__.co_varnames)
                ACTIVATION_FUNCTIONS[opt_name.lower()]["func_args"] = args

    return ACTIVATION_FUNCTIONS


def return_activation(activation_str):
    avail_acts = return_available_activations()
    try:
        # NOTE: this feels like the wrong place to add a .lower()
        activation = avail_acts[activation_str.lower()]
    except KeyError:
        raise KeyError(
            f"activation {activation_str} not available in options {avail_acts.keys()}"
        )

    return activation
