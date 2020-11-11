import inspect

import tensorflow as tf


class NOTPRESENT:
    def __init__(self):
        pass


# standard_args = {"opts": None, "name": None, "logger": None, "g_logger": None}

# TOOD: could this be automated?


def return_available_layers():
    # logic to get all layers in a class
    LAYER_FUNCTIONS = {}
    available_keras_layers = tf.keras.layers.__dict__
    for layer_name, layer_func in available_keras_layers.items():
        if inspect.isclass(layer_func):
            if issubclass(layer_func, tf.keras.layers.Layer):
                LAYER_FUNCTIONS[layer_name.lower()] = {}
                LAYER_FUNCTIONS[layer_name.lower()]["function"] = layer_func

    # LAYER_FUNCTIONS["custom_layer"] = {}
    # LAYER_FUNCTIONS["custom_layer"]["function"] = None
    # TODO: could add "func_args" here as desired

    return LAYER_FUNCTIONS


# def get_default_args(func):
#     spec = inspect.getfullargspec(func)
#     return (spec.args, spec.defaults)


def get_layer_options(layer_func):

    # I'm assuming there is an easier way to handle this
    # some_vars = get_default_args(layer_func)
    # print(some_vars)

    try:
        layer_opt_spec = inspect.getfullargspec(layer_func)
        cur_func_vars = layer_opt_spec.args

        # VERSION : > tf 2.2, NOTE: unsure about this being the "right" way to hand this
        if cur_func_vars == ["cls"]:
            layer_opt_spec = inspect.getfullargspec(layer_func.__init__)
            cur_func_vars = layer_opt_spec.args

        # else:
        # some layers don't have defaults (like reshape)
        if layer_opt_spec.defaults:
            cur_func_defaults = list(layer_opt_spec.defaults)
        else:
            cur_func_defaults = []
    except KeyError:
        # some layers inherit "__init__" from a base class e.g. batchnorm
        # the assumption here is that the 1st base class will contain the init..
        # I doubt this is accurate, but it is currently working
        try:
            first_parent = layer_func.__bases__[0]
            layer_opt_spec = inspect.getfullargspec(first_parent)
            cur_func_vars = layer_opt_spec.args
            # some layers don't have defaults (like reshape)
            if layer_opt_spec.defaults:
                cur_func_defaults = list(layer_opt_spec.defaults)
            else:
                cur_func_defaults = []
        except KeyError:
            raise NotImplementedError(
                f"error with type:{layer_func}, first parent: {first_parent}, other parents ({func.__bases__}). This error may be a result of an assumption that the __init__ params are from {first_parent} and not one of ({func.__bases__})"
            )
    if issubclass(layer_func, tf.keras.layers.Layer):
        # replace "kwargs"
        # cur_func_vars.remove("kwargs")
        # TODO: replace this with automated -- where did I get these from? # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer
        cur_func_vars.extend(["trainable", "name", "dtype", "dynamic"])
        cur_func_defaults.extend([True, None, None, False])

    try:
        cur_func_vars.remove("self")
    except ValueError:
        pass

    # VERSION : > tf 2.2
    try:
        cur_func_vars.remove("cls")
    except ValueError:
        pass

    diff = len(cur_func_vars) - len(cur_func_defaults)
    diff_l = [NOTPRESENT] * diff
    cur_func_defaults = diff_l + cur_func_defaults

    # sanity check
    assert len(cur_func_vars) == len(
        cur_func_defaults
    ), f"different number of defaults ({len(cur_func_vars)}) than allowed variables ({len(cur_func_defaults)}) defaults:{cur_func_defaults}, vars:{cur_func_vars}"

    return (cur_func_vars, cur_func_defaults)


def return_layer_defaults(layer_type):
    # logic to get all layers in a class
    if isinstance(layer_type, str):
        LAYER_FUNCTIONS = return_available_layers()
        if layer_type in LAYER_FUNCTIONS.keys():
            func = LAYER_FUNCTIONS[layer_type]["function"]
        else:
            raise ValueError(
                f"layer type {layer_type} not available in {LAYER_FUNCTIONS.keys()}"
            )
    elif callable(layer_type):
        func = layer_type
    else:
        raise ValueError(
            f"passed layer type is neither a string nor a module. {layer_type} is {type(layer_type)}"
        )

    func_args, func_defaults = get_layer_options(func)

    return {"func": func, "func_args": func_args, "func_defaults": func_defaults}
