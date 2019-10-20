import inspect

import tensorflow as tf

from yeahml.build.components.util import copy_func


def _configure_regularizer(opt_dict):
    # TODO: this is dangerous.... (updating the __defaults__ like this)
    reg_fn = return_regularizer(opt_dict["type"])["function"]
    reg_fn = copy_func(reg_fn)
    temp_copy = opt_dict.copy()
    _ = temp_copy.pop("type")
    if temp_copy:
        var_list = list(reg_fn.__code__.co_varnames)
        cur_defaults_list = list(reg_fn.__defaults__)
        for ao, v in temp_copy.items():
            try:
                arg_index = var_list.index(ao)
                cur_defaults_list[arg_index] = v
            except ValueError:
                raise ValueError(f"regularizer option {ao} not in options: {var_list}")
            # TODO: same type assertion?
        reg_fn.__defaults__ = tuple(cur_defaults_list)

    return reg_fn


def return_available_regularizers():

    REGULARIZER_FUNCTIONS = {}
    available_keras_regularizers = tf.keras.regularizers.__dict__
    for opt_name, opt_func in available_keras_regularizers.items():
        if callable(opt_func) and not inspect.isclass(opt_func):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                REGULARIZER_FUNCTIONS[opt_name.lower()] = {}
                REGULARIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = list(opt_func.__code__.co_varnames)
                REGULARIZER_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return REGULARIZER_FUNCTIONS


def return_regularizer(regularizer_str):
    avail_regularizers = return_available_regularizers()
    try:
        regularizer = avail_regularizers[regularizer_str]
    except KeyError:
        raise KeyError(
            f"regularizer {regularizer_str} not available in options {avail_regularizers.keys()}"
        )

    return regularizer
