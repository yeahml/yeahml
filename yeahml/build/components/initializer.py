import inspect

import tensorflow as tf


def _configure_initializer(opt_dict):
    # NOTE: I'm not thrilled with the initializer logic. the core issue is that
    # w/in tf2 there are both functions and classes implementing optimizers and
    # I'm not sure which I want to implement (both?) and have decided to implement
    # the class version --- however, I will need to think about how to implement
    # custom functions here

    try:
        cur_type = opt_dict["type"]
    except TypeError:
        # TODO: could include more helpful message here if the type is an initializer option
        raise TypeError(
            f"config for initialier does not specify a 'type'. Current specified options:({opt_dict})."
        )
    init_obj = return_initializer(cur_type.lower())
    init_fn = init_obj["function"]
    if not set(opt_dict["options"].keys()).issubset(init_obj["func_args"]):
        raise ValueError(
            f"options {opt_dict['options'].keys()} not in {init_obj['func_args']}"
        )

    # arg_order = vars(init_fn)["__init__"].__code__.co_varnames
    # default_values = vars(init_fn)["__init__"].__defaults__
    cur_config = init_fn().get_config()
    for k, v in opt_dict["options"].items():
        cur_config[k] = v
    out = init_fn().from_config(cur_config)

    return out


def return_available_initializers():

    # I don't feel great about this logic

    INITIALIZER_FUNCTIONS = {}
    available_keras_initializers = tf.keras.initializers.__dict__

    for opt_name, opt_func in available_keras_initializers.items():
        if inspect.isclass(opt_func) and issubclass(
            opt_func, tf.keras.initializers.Initializer
        ):  # callable(opt_func):  # or
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                INITIALIZER_FUNCTIONS[opt_name.lower()] = {}
                INITIALIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                try:
                    # args = opt_func().get_config()
                    args = list(vars(opt_func)["__init__"].__code__.co_varnames)
                    args = [a for a in args if a != "self"]
                except KeyError:
                    args = None

                INITIALIZER_FUNCTIONS[opt_name.lower()]["func_args"] = args

    return INITIALIZER_FUNCTIONS


def return_initializer(initializer_str: str):
    avail_initializers = return_available_initializers()
    try:
        # NOTE: this feels like the wrong place to add a .lower()
        initializer = avail_initializers[initializer_str.lower()]
    except KeyError:
        raise KeyError(
            f"initializer {initializer_str} not available in options {avail_initializers.keys()}"
        )

    return initializer
