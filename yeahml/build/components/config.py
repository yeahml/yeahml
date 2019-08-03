import inspect
import tensorflow as tf

# TODO: these two functions are repeatable and could be abstracted for each 'layer', 'optimizer', etc..


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


def return_available_activations():
    # logic to get all layers in a class
    ACTIVATION_FUNCTIONS = {}
    available_keras_activations = tf.keras.activations.__dict__

    for opt_name, opt_func in available_keras_activations.items():
        if callable(opt_func):
            # TODO: likely a better way to handle this.. but I'm unsure how personally
            if opt_name not in set(["deserialize", "get", "serialize"]):
                ACTIVATION_FUNCTIONS[opt_name.lower()] = {}
                ACTIVATION_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = list(opt_func.__code__.co_varnames)
                ACTIVATION_FUNCTIONS[opt_name.lower()]["func_args"] = args

    return ACTIVATION_FUNCTIONS


def return_activation(activation_str):
    avail_acts = return_available_activations()
    try:
        activation = avail_acts[activation_str]
    except KeyError:
        raise KeyError(
            f"activation {activation_str} not available in options {avail_acts.keys()}"
        )

    return activation


def return_available_regularizers():

    REGULARIZER_FUNCTIONS = {}
    available_keras_regularizers = tf.keras.regularizers.__dict__

    for opt_name, opt_func in available_keras_regularizers.items():
        if callable(opt_func) and not inspect.isclass(opt_func):
            if opt_name not in set(["deserialize", "get", "serialize"]):
                REGULARIZER_FUNCTIONS[opt_name.lower()] = {}
                REGULARIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = list(opt_func.__code__.co_varnames)
                REGULARIZER_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return REGULARIZER_FUNCTIONS
    # print(tf.keras.regularizers.__dict__)


def return_regularizer(regularizer_str):
    avail_regularizers = return_available_regularizers()
    try:
        regularizer = avail_regularizers[regularizer_str]
    except KeyError:
        raise KeyError(
            f"activation {regularizer_str} not available in options {avail_regularizers.keys()}"
        )

    return regularizer
