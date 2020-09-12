import inspect

import tensorflow as tf


def configure_regularizer(func_type, func_opt):
    reg_obj = return_regularizer(func_type)

    reg_fn = reg_obj["function"]
    cur_config = reg_fn().get_config()
    if func_opt:
        if not set(func_opt.keys()).issubset(reg_obj["func_args"]):
            raise ValueError(f"options {func_opt.keys()} not in {reg_obj['func_args']}")
        for k, v in func_opt.items():
            cur_config[k] = v
        out = reg_fn().from_config(cur_config)
    else:
        out = reg_fn()

    return out


def return_available_regularizers():
    REGULARIZER_FUNCTIONS = {}
    available_keras_regularizers = tf.keras.regularizers.__dict__
    for opt_name, opt_func in available_keras_regularizers.items():
        if inspect.isclass(opt_func) and issubclass(
            opt_func, tf.keras.regularizers.Regularizer
        ):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                REGULARIZER_FUNCTIONS[opt_name.lower()] = {}
                REGULARIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = inspect.signature(opt_func).parameters
                args = [a for a in args if a not in ["kwargs"]]
                REGULARIZER_FUNCTIONS[opt_name.lower()]["func_args"] = args

    return REGULARIZER_FUNCTIONS


def return_regularizer(regularizer_str):
    avail_regularizers = return_available_regularizers()
    try:
        # NOTE: this feels like the wrong place to add a .lower()
        regularizer = avail_regularizers[regularizer_str.lower()]
    except KeyError:
        raise KeyError(
            f"regularizer {regularizer_str} not available in options {avail_regularizers.keys()}"
        )

    return regularizer
