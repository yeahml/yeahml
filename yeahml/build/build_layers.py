import tensorflow as tf
import numpy as np
import sys
from typing import Any, List
from yeahml.build.components.activation import get_activation_fn
from yeahml.log.yf_logging import config_logger  # custom logging

# pooling layers
from yeahml.build.layers.pooling import build_pooling_layer

# conv layers
from yeahml.build.layers.convolution import build_conv_layer
from yeahml.build.layers.deconvolution import build_deconv_layer
from yeahml.build.layers.recurrent import build_recurrent_layer
from yeahml.build.layers.dense import build_dense_layer
from yeahml.build.layers.other import (
    build_embedding_layer,
    build_batch_normalization_layer,
)

# dropout
from yeahml.build.layers.dropout import build_dropout_layer


def build_hidden_block(MCd: dict, HCd: dict, logger, g_logger) -> List[Any]:
    # logger = config_logger(MCd, "build")
    logger.info("-> START building hidden layers")
    HIDDEN_LAYERS = []

    # build each layer based on the (ordered) yaml specification
    logger.debug("loop+start building layers: {}".format(HCd["layers"].keys()))
    for i, l_name in enumerate(HCd["layers"]):
        layer_info = HCd["layers"][str(l_name)]
        logger.debug(
            "-> START building layer: {} with opts: {}".format(l_name, layer_info)
        )

        # TODO: remove this assignment (already checked in config)
        opts = layer_info["options"]
        actfn = get_activation_fn(layer_info["activation"])

        ltype = layer_info["type"].lower()
        logger.debug(f"-> START building: {ltype} - {l_name}")
        if ltype == "conv":
            cur_layer = build_conv_layer(opts, actfn, l_name, logger, g_logger)
        elif ltype == "deconv" or ltype == "conv_transpose":  # TODO: simplify
            cur_input = build_deconv_layer(opts, actfn, l_name, logger, g_logger)
        elif ltype == "flatten":
            cur_layer = tf.keras.layers.Flatten()
        elif ltype == "dense":
            # TODO: need to flatten?
            cur_layer = build_dense_layer(opts, actfn, l_name, logger, g_logger)
        # TODO: consolidate pooling layers
        elif ltype == "pooling":
            cur_layer = build_pooling_layer(opts, l_name, logger, g_logger)
        elif ltype == "embedding":
            # cur_input = build_embedding_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "batch_normalization":
            # cur_input = build_batch_normalization_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "recurrent":
            # cur_input = build_recurrent_layer(
            #     cur_input, training, opts, actfn, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "dropout":
            cur_layer = build_dropout_layer(opts, l_name, logger, g_logger)
        else:
            raise NotImplementedError(f"layer type: {ltype} not implemented yet")

        HIDDEN_LAYERS.append(cur_layer)

    logger.info("[END] building hidden block")

    return HIDDEN_LAYERS
