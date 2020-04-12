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
    except KeyError:
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
    LAYER_FUNCTIONS = return_available_layers()
    if layer_type in LAYER_FUNCTIONS.keys():
        func = LAYER_FUNCTIONS[layer_type]["function"]
    else:
        raise ValueError(
            f"layer type {layer_type} not available in {LAYER_FUNCTIONS.keys()}"
        )

    func_args, func_defaults = get_layer_options(func)

    return {"func": func, "func_args": func_args, "func_defaults": func_defaults}


# LAYER_FUNCTIONS = {}
# # Convolutions
# LAYER_FUNCTIONS["Conv1D".lower()] = {"function": tf.keras.layers.Conv1D}
# LAYER_FUNCTIONS["Conv2D".lower()] = {"function": tf.keras.layers.Conv2D}
# LAYER_FUNCTIONS["Conv3D".lower()] = {"function": tf.keras.layers.Conv3D}
# LAYER_FUNCTIONS["Dense".lower()] = {"function": tf.keras.layers.Dense}

# # conv transpose
# LAYER_FUNCTIONS["Conv2DTranspose".lower()] = {
#     "function": tf.keras.layers.Conv2DTranspose
# }
# LAYER_FUNCTIONS["Conv3DTranspose".lower()] = {
#     "function": tf.keras.layers.Conv3DTranspose
# }

# # dropout
# LAYER_FUNCTIONS["Dropout".lower()] = {"function": tf.keras.layers.Dropout}
# LAYER_FUNCTIONS["AlphaDropout".lower()] = {"function": tf.keras.layers.AlphaDropout}
# LAYER_FUNCTIONS["GaussianDropout".lower()] = {
#     "function": tf.keras.layers.GaussianDropout
# }
# LAYER_FUNCTIONS["SpatialDropout1D".lower()] = {
#     "function": tf.keras.layers.SpatialDropout1D
# }
# LAYER_FUNCTIONS["SpatialDropout2D".lower()] = {
#     "function": tf.keras.layers.SpatialDropout2D
# }
# LAYER_FUNCTIONS["SpatialDropout3D".lower()] = {
#     "function": tf.keras.layers.SpatialDropout3D
# }


# # pooling
# LAYER_FUNCTIONS["GlobalAveragePooling1D".lower()] = {
#     "function": tf.keras.layers.GlobalAveragePooling1D
# }
# LAYER_FUNCTIONS["GlobalAveragePooling2D".lower()] = {
#     "function": tf.keras.layers.GlobalAveragePooling2D
# }
# LAYER_FUNCTIONS["GlobalAveragePooling3D".lower()] = {
#     "function": tf.keras.layers.GlobalAveragePooling3D
# }

# LAYER_FUNCTIONS["AveragePooling1D".lower()] = {
#     "function": tf.keras.layers.AveragePooling1D
# }
# LAYER_FUNCTIONS["AveragePooling2D".lower()] = {
#     "function": tf.keras.layers.AveragePooling2D
# }
# LAYER_FUNCTIONS["AveragePooling3D".lower()] = {
#     "function": tf.keras.layers.AveragePooling3D
# }

# LAYER_FUNCTIONS["GlobalMaxPool1D".lower()] = {
#     "function": tf.keras.layers.GlobalMaxPool1D
# }
# LAYER_FUNCTIONS["GlobalMaxPool2D".lower()] = {
#     "function": tf.keras.layers.GlobalMaxPool2D
# }
# LAYER_FUNCTIONS["GlobalMaxPool3D".lower()] = {
#     "function": tf.keras.layers.GlobalMaxPool3D
# }

# LAYER_FUNCTIONS["MaxPool1D".lower()] = {"function": tf.keras.layers.MaxPool1D}
# LAYER_FUNCTIONS["MaxPool2D".lower()] = {"function": tf.keras.layers.MaxPool2D}
# LAYER_FUNCTIONS["MaxPool3D".lower()] = {"function": tf.keras.layers.MaxPool3D}


# # others
# LAYER_FUNCTIONS["flatten"] = {"function": tf.keras.layers.Flatten}
