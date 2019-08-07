import inspect
import tensorflow as tf


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
