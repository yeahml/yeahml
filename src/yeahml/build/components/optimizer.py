import inspect

import tensorflow as tf


def configure_optimizer(opt_dict):
    # TODO: this should not be here. (follow template for losses)
    optim_dict = return_optimizer(opt_dict["type"])
    optimizer = optim_dict["function"]

    # configure optimizers
    temp_dict = opt_dict.copy()
    optimizer = optimizer(**temp_dict["options"])

    return optimizer


def return_available_optimizers():
    # logic to get all layers in a class
    OPTIMIZER_FUNCTIONS = {}
    available_keras_optimizers = tf.keras.optimizers.__dict__
    for opt_name, opt_func in available_keras_optimizers.items():
        # TODO: could change to is subclass tf.keras.optimizers.Optimizer
        if opt_name.lower() != "optimizer":  # NOTE: hardcoded
            if inspect.isclass(opt_func):
                OPTIMIZER_FUNCTIONS[opt_name.lower()] = {}
                OPTIMIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func

                args = list(vars(opt_func)["__init__"].__code__.co_varnames)
                filt_args = [a for a in args if a != "self"]
                OPTIMIZER_FUNCTIONS[opt_name.lower()]["func_args"] = filt_args

    return OPTIMIZER_FUNCTIONS


def return_optimizer(optimizer_str):
    avail_opts = return_available_optimizers()
    try:
        optimizer = avail_opts[optimizer_str]
    except KeyError:
        raise KeyError(
            f"optimizer {optimizer_str} not available in options {avail_opts.keys()}"
        )

    return optimizer
