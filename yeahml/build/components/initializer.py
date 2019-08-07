import inspect
import tensorflow as tf


def return_available_initializers():

    # I don't feel great about this logic

    INITIALIZER_FUNCTIONS = {}
    available_keras_initializers = tf.keras.initializers.__dict__

    # print(available_keras_initializers.keys())

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
        # elif callable(opt_func):
        #     if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
        #         INITIALIZER_FUNCTIONS[opt_name.lower()] = {}
        #         INITIALIZER_FUNCTIONS[opt_name.lower()]["function"] = opt_func
        #         args = list(opt_func.__code__.co_varnames)
        #         INITIALIZER_FUNCTIONS[opt_name.lower()]["func_args"] = args

    return INITIALIZER_FUNCTIONS


def return_initializer(initializer_str):
    avail_initializers = return_available_initializers()
    try:
        initializer = avail_initializers[initializer_str]
    except KeyError:
        raise KeyError(
            f"activation {initializer_str} not available in options {avail_initializers.keys()}"
        )

    return initializer
