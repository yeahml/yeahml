import inspect

import tensorflow as tf

from yeahml.build.components.util import copy_func


def configure_activation(func_type, func_opt):
    act_fn = return_activation(func_type)["function"]

    if func_opt:
        temp_copy = func_opt.copy()

        if "__original_wrapped__" in act_fn.__dict__:
            act_fn = copy_func(act_fn.__dict__["__original_wrapped__"])
        else:
            act_fn = copy_func(act_fn)

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
                args = inspect.signature(opt_func).parameters
                args = [a for a in args if a not in ["x"]]
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
