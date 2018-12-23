import tensorflow as tf
import numpy as np
import sys
from yeahml.build.get_components import get_activation_fn
from yeahml.log.yf_logging import config_logger  # custom logging

# pooling layers
from yeahml.build.layers.pooling import (
    build_global_pooling_layer,
    build_pool_1d_layer,
    build_pool_2d_layer,
)

# conv layers
from yeahml.build.layers.convolution import (
    build_conv2d_transpose_layer,
    build_conv2d_layer,
)
from yeahml.build.layers.recurrent import build_recurrent_layer
from yeahml.build.layers.dense import build_dense_layer
from yeahml.build.layers.other import (
    build_embedding_layer,
    build_batch_normalization_layer,
)


def build_hidden_block(X, training, MCd: dict, HCd: dict, logger, g_logger):
    # logger = config_logger(MCd, "build")
    logger.info("-> START building hidden block")
    # in: X
    # out: last layer before logits

    # cur_input will be updated each iteration such that on
    # the next iteration it will be the input to the following layer
    cur_input = X
    control_deps = []

    # build each layer based on the (ordered) yaml specification
    logger.debug("loop+start building layers: {}".format(HCd["layers"].keys()))
    for i, l_name in enumerate(HCd["layers"]):
        layer_info = HCd["layers"][str(l_name)]
        logger.debug(
            "-> START building layer: {} with opts: {}".format(l_name, layer_info)
        )

        try:
            opts = layer_info["options"]
        except:
            opts = None
            pass

        try:
            actfn_str = layer_info["activation"]
            actfn = get_activation_fn(actfn_str)
        except KeyError:
            try:
                actfn = get_activation_fn(MCd["def_act"])
            except KeyError:
                # the reasoning here is that the relu is subjectively the most
                # common/default activation function in DNNs, but I don't LOVE this
                actfn = get_activation_fn("relu")
        logger.debug("activation set: {}".format(actfn))

        ltype = layer_info["type"].lower()
        if ltype == "conv2d":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_conv2d_layer(
                cur_input, opts, actfn, l_name, logger, g_logger
            )
        elif ltype == "deconv2d":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_conv2d_transpose_layer(
                cur_input, opts, actfn, l_name, logger, g_logger
            )
        elif ltype == "dense":
            logger.debug("-> START building: {}".format(ltype))
            # ---- this block is 'dumb' but works --------------
            # this is necessary because if the last block was a pool
            # or conv, we need to flatten layer before we add a dense layer
            prev_ltype_key = list(HCd["layers"])[i - 1]
            prev_ltype = HCd["layers"][prev_ltype_key]["type"]
            if (
                prev_ltype == "conv2d"
                or prev_ltype == "pooling2d"
                # or prev_ltype == "pooling1d"
            ):
                # flatten
                # maybe remove np dependency cur_input.get_shape().as_list()[1:]
                last_shape = int(np.prod(cur_input.get_shape()[1:]))
                cur_input = tf.reshape(cur_input, shape=[-1, last_shape])
                logger.debug("reshaped tensor: {}".format(cur_input))
                g_logger.info(">> flatten: {}".format(cur_input.shape))
            # --------------------------------------------------
            cur_input = build_dense_layer(
                cur_input, training, opts, actfn, l_name, logger, g_logger
            )
        elif ltype == "pooling2d":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_pool_2d_layer(
                cur_input, training, opts, l_name, logger, g_logger
            )
        elif ltype == "pooling1d":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_pool_1d_layer(
                cur_input, training, opts, l_name, logger, g_logger
            )
        elif ltype == "global_pooling":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_global_pooling_layer(
                cur_input, training, opts, l_name, logger, g_logger
            )
        elif ltype == "embedding":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_embedding_layer(
                cur_input, training, opts, l_name, logger, g_logger
            )
        elif ltype == "batch_normalization":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_batch_normalization_layer(
                cur_input, training, opts, l_name, logger, g_logger
            )
            control_deps.append("batch_norm ({})".format(HCd["layers"][str(l_name)]))
        elif ltype == "recurrent":
            logger.debug("-> START building: {}".format(ltype))
            cur_input = build_recurrent_layer(
                cur_input, training, opts, actfn, l_name, logger, g_logger
            )
        else:
            logger.fatal("unable to build layer type: {}".format(ltype))
            sys.exit("unable to build layer type: {}".format(ltype))

    logger.info("[END] building hidden block")

    return cur_input, control_deps
