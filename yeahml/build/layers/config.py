from yeahml.build.layers.other import build_flatten_layer
import inspect
import tensorflow as tf

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

