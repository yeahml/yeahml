import tensorflow as tf
import numpy as np
import sys
from typing import Any, List
from yeahml.build.components.activation import get_activation_fn
from yeahml.log.yf_logging import config_logger  # custom logging

# pooling layers
from yeahml.build.layers.pooling import (
    build_global_pooling_layer,
    build_pool_1d_layer,
    build_pool_2d_layer,
)

# conv layers
from yeahml.build.layers.convolution import build_conv_layer
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
        # if ltype == "conv2d":
        #     logger.debug("-> START building: {}".format(ltype))
        #     cur_layer = build_conv2d_layer(opts, actfn, l_name, logger, g_logger)
        if ltype == "deconv2d":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_conv2d_transpose_layer(
            #     cur_input, opts, actfn, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "conv":
            cur_layer = build_conv_layer(opts, actfn, l_name, logger, g_logger)
        elif ltype == "flatten":
            logger.debug("-> START building: {}".format(ltype))
            cur_layer = tf.keras.layers.Flatten()
        elif ltype == "dense":
            logger.debug("-> START building: {}".format(ltype))
            # TODO: need to flatten?
            cur_layer = build_dense_layer(opts, actfn, l_name, logger, g_logger)
        elif ltype == "pooling2d":
            logger.debug("-> START building: {}".format(ltype))
            cur_layer = build_pool_2d_layer(opts, l_name, logger, g_logger)
        elif ltype == "pooling1d":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_pool_1d_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "global_pooling":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_global_pooling_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "embedding":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_embedding_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "batch_normalization":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_batch_normalization_layer(
            #     cur_input, training, opts, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "recurrent":
            logger.debug("-> START building: {}".format(ltype))
            # cur_input = build_recurrent_layer(
            #     cur_input, training, opts, actfn, l_name, logger, g_logger
            # )
            raise NotImplementedError
        elif ltype == "dropout":
            logger.debug("-> START building: {}".format(ltype))
            cur_layer = build_dropout_layer(opts, l_name, logger, g_logger)
        else:
            logger.fatal("unable to build layer type: {}".format(ltype))
            sys.exit("unable to build layer type: {}".format(ltype))

        HIDDEN_LAYERS.append(cur_layer)

    logger.info("[END] building hidden block")

    return HIDDEN_LAYERS
