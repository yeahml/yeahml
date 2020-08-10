import inspect

import tensorflow as tf


def configure_initializer(func_type, func_opt):
    init_obj = return_initializer(func_type)

    init_fn = init_obj["function"]
    cur_config = init_fn().get_config()
    if func_opt:
        if not set(func_opt.keys()).issubset(init_obj["func_args"]):
            raise ValueError(
                f"options {func_opt.keys()} not in {init_obj['func_args']}"
            )
        for k, v in func_opt.items():
            cur_config[k] = v
        out = init_fn().from_config(cur_config)
    else:
        out = init_fn()

    return out


def return_available_initializers():

    # I don't feel great about this logic

    INITIALIZER_FUNCTIONS = {}
    available_keras_initializers = tf.keras.initializers.__dict__

    for opt_name, opt_func in available_keras_initializers.items():
        if inspect.isclass(opt_func) and issubclass(
            opt_func, tf.keras.initializers.Initializer
        ):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                INITIALIZER_FUNCTIONS[opt_name.lower()] = {}
                INITIALIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = inspect.signature(opt_func).parameters
                args = list(args)
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
