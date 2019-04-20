import tensorflow as tf
import sys


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
            out = tf.reduce_mean(cur_input, [1, 2], name=name)
    else:
        sys.exit("pool type {} is not allowed".format(pool_type))
    logger.debug("tensor obj: {}".format(out))
    # g_logger.info("{}".format(fmt_tensor_info(out)))
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
    # g_logger.info("{}".format(fmt_tensor_info(out)))

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


def build_pool_2d_layer(cur_input, training, opts: dict, name: str, logger, g_logger):

    # TODO: this config should be moved to the config
    try:
        if opts:
            pool_size = opts["pool_size"]
        else:
            pool_size = [2, 2]
    except KeyError:
        pool_size = [2, 2]
    logger.debug("pool_size set: {}".format(pool_size))

    # TODO: this config should be moved to the config
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
    # g_logger.info("{}".format(fmt_tensor_info(out)))

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
