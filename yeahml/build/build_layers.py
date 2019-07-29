import tensorflow as tf
import numpy as np
import sys
from typing import Any, List
from yeahml.build.components.activation import get_activation_fn
from yeahml.log.yf_logging import config_logger  # custom logging

# pooling layers
# from yeahml.build.layers.pooling import build_pooling_layer

# conv layers
# from yeahml.build.layers.convolution import build_conv_layer
from yeahml.build.layers.deconvolution import build_deconv_layer

from yeahml.build.layers.recurrent import build_recurrent_layer

# from yeahml.build.layers.dense import build_dense_layer
from yeahml.build.layers.config import LAYER_FUNCTIONS

from yeahml.build.layers.other import (
    build_embedding_layer,
    build_batch_normalization_layer,
)

# dropout
# from yeahml.build.layers.dropout import build_dropout_layer


def build_layer(ltype, opts, actfn, l_name, logger, g_logger):

    # # TODO: this needs to be pushed down a level since some layers doesn't require act
    # HLD["options"].update({"activation": None})
    # try:
    #     actfn_str = hl["activation"]
    # except KeyError:
    #     try:
    #         # fall back to the default activation function specified
    #         actfn_str = default_activation
    #     except KeyError:
    #         # relu: the reasoning here is that the relu is subjectively the most
    #         # common/default activation function in DNNs, but I don't LOVE this
    #         actfn_str = "relu"
    # # TODO: logger.debug("activation set: {}".format(actfn_str))
    # HLD["options"]["activation"] = actfn_str
    # TODO: could place a ltype = get_ltype_mapping() here
    if ltype in LAYER_FUNCTIONS.keys():
        func = LAYER_FUNCTIONS[ltype]["function"]
        try:
            func_args = LAYER_FUNCTIONS[ltype]["func_args"]
            opts.update(func_args)
        except KeyError:
            pass
        print(opts)

        # .__code__.co_varnames provides a lits of func arguments
        if "actfn" in func.__code__.co_varnames:
            print("SUUUUUPPPPPPPP")
            cur_layer = func(opts, actfn, l_name, logger, g_logger)
        else:
            cur_layer = func(opts, l_name, logger, g_logger)

    return cur_layer


def build_hidden_block(MCd: dict, HCd: dict, logger, g_logger) -> List[Any]:
    # logger = config_logger(MCd, "build")
    logger.info("-> START building hidden layers")
    HIDDEN_LAYERS = []

    print(HCd)

    # build each layer based on the (ordered) yaml specification
    logger.debug(f"loop+start building layers: {HCd['layers'].keys()}")
    for i, l_name in enumerate(HCd["layers"]):
        layer_info = HCd["layers"][str(l_name)]
        logger.debug(f"-> START building layer: {l_name} with opts: {layer_info}")

        # TODO: remove this assignment (already checked in config)
        opts = layer_info["options"]
        actfn = get_activation_fn(layer_info["activation"])

        ltype = layer_info["type"].lower()
        logger.debug(f"-> START building: {ltype} - {l_name}")
        cur_layer = build_layer(ltype, opts, actfn, l_name, logger, g_logger)
        # if ltype == "conv":
        #     cur_layer = build_conv_layer(opts, actfn, l_name, logger, g_logger)
        # elif ltype == "deconv" or ltype == "conv_transpose":  # TODO: simplify
        #     cur_input = build_deconv_layer(opts, actfn, l_name, logger, g_logger)
        # elif ltype == "flatten":
        #     cur_layer = tf.keras.layers.Flatten()
        # elif ltype == "dense":
        #     # TODO: need to flatten?
        #     cur_layer = build_dense_layer(opts, actfn, l_name, logger, g_logger)
        # # TODO: consolidate pooling layers
        # elif ltype == "pooling":
        #     cur_layer = build_pooling_layer(opts, l_name, logger, g_logger)
        # elif ltype == "embedding":
        #     cur_layer = build_embedding_layer(opts, l_name, logger, g_logger)
        # elif ltype == "batch_normalization":
        #     cur_layer = build_batch_normalization_layer(opts, l_name, logger, g_logger)
        # elif ltype == "recurrent":
        #     cur_layer = build_recurrent_layer(opts, actfn, l_name, logger, g_logger)
        # elif ltype == "dropout":
        #     cur_layer = build_dropout_layer(opts, l_name, logger, g_logger)
        # else:
        #     raise NotImplementedError(f"layer type: {ltype} not implemented yet")

        HIDDEN_LAYERS.append(cur_layer)

    logger.info("[END] building hidden block")

    return HIDDEN_LAYERS
