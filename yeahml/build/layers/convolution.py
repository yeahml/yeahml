import tensorflow as tf
from yeahml.build.get_components import (
    get_regularizer_fn,
    get_initializer_fn,
    get_activation_fn,
)
from yeahml.helper import fmt_tensor_info

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
