from yeahml.build.layers.convolution import build_conv_layer
from yeahml.build.layers.dense import build_dense_layer
from yeahml.build.layers.dropout import build_dropout_layer
from yeahml.build.layers.pooling import build_pooling_layer
from yeahml.build.layers.other import build_flatten_layer

import tensorflow as tf

standard_args = {"opts": None, "name": None, "logger": None, "g_logger": None}

# TODO: specify input

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS["conv2d"] = {"function": tf.keras.layers.Conv2D}
LAYER_FUNCTIONS["dense"] = {"function": tf.keras.layers.Dense}
LAYER_FUNCTIONS["dropout"] = {"function": tf.keras.layers.Dropout}
LAYER_FUNCTIONS["averagepooling2d"] = {"function": tf.keras.layers.AveragePooling2D}
LAYER_FUNCTIONS["flatten"] = {"function": tf.keras.layers.Flatten}

