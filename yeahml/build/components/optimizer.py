import inspect
import tensorflow as tf


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
            f"optimzer {optimizer_str} not available in options {avail_opts.keys()}"
        )

    return optimizer
