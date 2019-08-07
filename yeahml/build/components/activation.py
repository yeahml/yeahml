import inspect
import tensorflow as tf


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
        activation = avail_acts[activation_str]
    except KeyError:
        raise KeyError(
            f"activation {activation_str} not available in options {avail_acts.keys()}"
        )

    return activation
