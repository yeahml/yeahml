from yeahml.build.layers.convolution import build_conv_layer
from yeahml.build.layers.dense import build_dense_layer
from yeahml.build.layers.dropout import build_dropout_layer
from yeahml.build.layers.pooling import build_pooling_layer
from yeahml.build.layers.other import build_flatten_layer

import tensorflow as tf

standard_args = {"opts": None, "name": None, "logger": None, "g_logger": None}

# TODO: specify input

LAYER_FUNCTIONS = {}
LAYER_FUNCTIONS["conv"] = {"function": build_conv_layer}
LAYER_FUNCTIONS["dense"] = {"function": build_dense_layer}
LAYER_FUNCTIONS["dropout"] = {"function": build_dropout_layer}
LAYER_FUNCTIONS["pooling"] = {"function": build_pooling_layer}
LAYER_FUNCTIONS["flatten"] = {"function": build_flatten_layer}

