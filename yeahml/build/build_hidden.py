import tensorflow as tf
import numpy as np
import sys

from yeahml.helper import fmt_tensor_info
from yeahml.build.get_components import (
    get_regularizer_fn,
    get_initializer_fn,
    get_activation_fn,
)
from yeahml.log.yf_logging import config_logger  # custom logging

# NOTE: the default padding is "same", this is different from the API which is "same"


def build_conv2d_transpose_layer(
    cur_input, opts: dict, actfn, name: str, logger, g_logger
):
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None
    logger.debug("k_init_fn set: {}".format(k_init_fn))

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None
    logger.debug("k_reg set: {}".format(k_reg))

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None
    logger.debug("b_reg set: {}".format(b_reg))

    filters = opts["filters"]
    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        kernel_size = 3
    logger.debug("kernel_size set: {}".format(kernel_size))

    try:
        # TODO: create func (w/error handling) for this
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"
    logger.debug("padding set: {}".format(padding))

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 1
    logger.debug("strides set: {}".format(strides))

    if not name:
        name = "unnamed_conv_transpose_layer"
    logger.debug("name set: {}".format(name))

    try:
        trainable = opts["trainable"]
    except KeyError:
        # trainable by default
        trainable = True
    logger.debug("trainable set: {}".format(trainable))

    out = tf.layers.conv2d_transpose(
        cur_input,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=actfn,
        use_bias=True,
        kernel_initializer=k_init_fn,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        trainable=trainable,
        name=name,
    )

    logger.debug("Final tensor obj: {}".format(out))

    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))
    return out


def build_conv2d_layer(cur_input, opts: dict, actfn, name: str, logger, g_logger):
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None
    logger.debug("k_init_fn set: {}".format(k_init_fn))

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None
    logger.debug("k_reg set: {}".format(k_reg))

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None
    logger.debug("b_reg set: {}".format(b_reg))

    filters = opts["filters"]
    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        kernel_size = 3
    logger.debug("kernel_size set: {}".format(kernel_size))

    try:
        # TODO: create func (w/error handling) for this
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"
    logger.debug("padding set: {}".format(padding))

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 1
    logger.debug("strides set: {}".format(strides))

    if not name:
        name = "unnamed_conv_layer"
    logger.debug("name set: {}".format(name))

    try:
        trainable = opts["trainable"]
    except KeyError:
        # trainable by default
        trainable = True
    logger.debug("trainable set: {}".format(trainable))

    out = tf.layers.conv2d(
        cur_input,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=actfn,
        kernel_initializer=k_init_fn,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        trainable=trainable,
        name=name,
    )
    logger.debug("Final tensor obj: {}".format(out))

    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))
    return out


def build_dense_layer(
    cur_input, training, opts: dict, actfn, name: str, logger, g_logger
):
    units = opts["units"]
    logger.debug("units set: {}".format(units))

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None
    logger.debug("k_init_fn set: {}".format(k_init_fn))

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None
    logger.debug("k_reg set: {}".format(k_reg))

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None
    logger.debug("b_reg set: {}".format(b_reg))

    try:
        trainable = opts["trainable"]
    except KeyError:
        # trainable by default
        trainable = True
    logger.debug("trainable set: {}".format(trainable))

    out = tf.layers.dense(
        inputs=cur_input,
        units=units,
        activation=actfn,
        kernel_initializer=k_init_fn,
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        trainable=trainable,
        name=name,
    )

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))

    ## add dropout, if indicated
    try:
        dropout_rate = opts["dropout"]
    except KeyError:
        dropout_rate = None
    logger.debug("dropout_rate set: {}".format(dropout_rate))

    if dropout_rate:
        out = tf.layers.dropout(
            inputs=out,
            rate=dropout_rate,
            noise_shape=None,
            seed=None,
            training=training,
            name=None,
        )
        logger.debug("tensor obj post dropout: {}".format(out))
        g_logger.info(">> dropout: {}".format(dropout_rate))

    logger.debug("[End] building: {}".format(name))
    return out


def build_pool_2d_layer(cur_input, training, opts: dict, name: str, logger, g_logger):

    try:
        if opts:
            pool_size = opts["pool_size"]
        else:
            pool_size = [2, 2]
    except KeyError:
        pool_size = [2, 2]
    logger.debug("pool_size set: {}".format(pool_size))

    try:
        if opts:
            strides = opts["strides"]
        else:
            strides = 2
    except KeyError:
        strides = 2
    logger.debug("strides set: {}".format(strides))

    if not name:
        name = "unnamed_pool2d_layer"
    logger.debug("name set: {}".format(name))

    try:
        if opts:
            pool_type = opts["pool_type"]
        else:
            pool_type = "max"
        if pool_type not in ["max", "avg"]:
            sys.exit("pool type {} is not allowed".format(pool_type))
    except KeyError:
        pool_type = "max"
    logger.debug("pool_type set: {}".format(pool_type))

    if pool_type == "max":
        out = tf.layers.max_pooling2d(
            cur_input, pool_size=pool_size, strides=strides, padding="same", name=name
        )
    elif pool_type == "avg":
        out = tf.layers.average_pooling2d(
            cur_input, pool_size=pool_size, strides=strides, padding="same", name=name
        )
    else:
        # for future pooling implementations
        sys.exit("pool type {} is not yet implemented".format(pool_type))

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))

    ## add dropout, if indicated
    try:
        if opts:
            dropout_rate = opts["dropout"]
        else:
            dropout_rate = None
    except KeyError:
        dropout_rate = None
    logger.debug("dropout_rate set: {}".format(dropout_rate))

    if dropout_rate:
        out = tf.layers.dropout(
            inputs=out,
            rate=dropout_rate,
            noise_shape=None,
            seed=None,
            training=training,
            name=None,
        )
        logger.debug("tensor obj post dropout: {}".format(out))
        g_logger.info(">> dropout: {}".format(dropout_rate))

    logger.debug("[End] building: {}".format(name))

    return out


def build_global_pooling_layer(
    cur_input, training, opts: dict, name: str, logger, g_logger
):

    # TODO: implement global max pooling
    try:
        if opts:
            pool_type = opts["type"]
        else:
            pool_type = "avg"
    except KeyError:
        pool_type = "avg"
    logger.debug("pool_type set: {}".format(pool_type))

    if pool_type == "avg":
        # TODO: this 2 is hard coded
        # in a 2d situation, this would be [1,2]
        if cur_input.get_shape().ndims == 3:
            out = tf.reduce_mean(cur_input, [2], name=name)
        elif cur_input.get_shape().ndims == 4:
            out = tf.reduce_mean(cur_input, [1,2], name=name)
    else:
        sys.exit("pool type {} is not allowed".format(pool_type))
    logger.debug("tensor obj: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))

    return out


def build_pool_1d_layer(cur_input, training, opts: dict, name: str, logger, g_logger):

    try:
        if opts:
            pool_size = opts["pool_size"]
        else:
            pool_size = 3
    except KeyError:
        pool_size = 3
    logger.debug("pool_size set: {}".format(pool_size))

    try:
        if opts:
            strides = opts["strides"]
        else:
            strides = 3
    except KeyError:
        strides = 3
    logger.debug("strides set: {}".format(strides))

    if not name:
        name = "unnamed_pool1d_layer"
    logger.debug("name set: {}".format(name))

    try:
        if opts:
            pool_type = opts["pool_type"]
        else:
            pool_type = "max"
        if pool_type not in ["max", "avg"]:
            sys.exit("pool type {} is not allowed".format(pool_type))
    except KeyError:
        pool_type = "max"
    logger.debug("pool_type set: {}".format(pool_type))

    if pool_type == "max":
        out = tf.layers.max_pooling1d(
            cur_input, pool_size=pool_size, strides=strides, padding="same", name=name
        )
    elif pool_type == "avg":
        out = tf.layers.average_pooling1d(
            cur_input, pool_size=pool_size, strides=strides, padding="same", name=name
        )
    else:
        # for future pooling implementations
        sys.exit("pool type {} is not yet implemented".format(pool_type))

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))

    ## add dropout, if indicated
    try:
        if opts:
            dropout_rate = opts["dropout"]
        else:
            dropout_rate = None
    except KeyError:
        dropout_rate = None
    logger.debug("dropout_rate set: {}".format(dropout_rate))

    if dropout_rate:
        out = tf.layers.dropout(
            inputs=out,
            rate=dropout_rate,
            noise_shape=None,
            seed=None,
            training=training,
            name=None,
        )
        logger.debug("tensor obj post dropout: {}".format(out))
        g_logger.info(">> dropout: {}".format(dropout_rate))

    logger.debug("[End] building: {}".format(name))

    return out


def build_embedding_layer(cur_input, training, opts: dict, name: str, logger, g_logger):

    # TODO: the default vocabulary_size size does not make sense here
    try:
        if opts:
            vocabulary_size = opts["vocabulary_size"]
        else:
            vocabulary_size = 100
    except KeyError:
        vocabulary_size = 100
    logger.debug("vocabulary_size set: {}".format(vocabulary_size))

    # TODO: the default embedding_size size does not make sense here
    try:
        if opts:
            embedding_size = opts["embedding_size"]
        else:
            embedding_size = 2
    except KeyError:
        embedding_size = 2
    logger.debug("embedding_size set: {}".format(embedding_size))

    if not name:
        name = "unnamed_embedding_layer"
    logger.debug("name set: {}".format(name))

    # TODO: I'm not sure this implemented correctly
    word_embeddings = tf.get_variable(
        "word_embeddings", [vocabulary_size, embedding_size]
    )
    embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, cur_input)

    out = embedded_word_ids

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))

    # TODO: VIZUALIZING EMBEDDINGS - https://www.tensorflow.org/guide/embedding

    return out


def build_hidden_block(X, training, MCd: dict, HCd: dict, logger, g_logger):
    # logger = config_logger(MCd, "build")
    logger.info("-> START building hidden block")
    # in: X
    # out: last layer before logits

    # cur_input will be updated each iteration such that on
    # the next iteration it will be the input to the following layer
    cur_input = X

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
        else:
            logger.fatal("unable to build layer type: {}".format(ltype))
            sys.exit("unable to build layer type: {}".format(ltype))

    logger.info("[END] building hidden block")

    return cur_input
