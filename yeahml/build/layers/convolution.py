import tensorflow as tf
from yeahml.build.get_components import get_initializer_fn
from yeahml.build.components.regularizer import get_regularizer_fn
from yeahml.build.components.activation import get_activation_fn
from typing import Any

from yeahml.helper import fmt_tensor_info

# NOTE: the default padding is "same", this is different from the API which is "same"


def build_conv_layer(opts: dict, actfn, name: str, logger, g_logger) -> Any:
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    # TODO: implement tf.keras.layers.DepthwiseConv2D

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
        data_format = opts["data_format"]
    except:
        data_format = None
    logger.debug("data_format set: {}".format(data_format))

    # try:
    #     dilation_rate = opts["dilation_rate"]
    # except:
    #     dilation_rate = (1, 1)
    # logger.debug("dilation_rate set: {}".format(dilation_rate))

    try:
        # TODO: create func (w/error handling) for this
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"
    logger.debug("padding set: {}".format(padding))

    if not name:
        name = "unnamed_conv_layer"
    logger.debug("name set: {}".format(name))

    # try:
    #     trainable = opts["trainable"]
    # except KeyError:
    #     # trainable by default
    #     trainable = True
    # logger.debug("trainable set: {}".format(trainable))

    try:
        conv_dim = opts["dimension"]  # TOOD: shorthand 'dim'
    except KeyError:
        raise ValueError(f"convolution dim not specified for {name}")

    # TODO: make sure these correspond to the current layer type
    try:
        filters = opts["filters"]
    except KeyError:
        raise ValueError(f"filters not set for {name}")
    logger.debug("filters set: {}".format(filters))

    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        raise ValueError(f"kernel_size not set for {name}")
    logger.debug(f"kernel_size set: {kernel_size}")

    try:
        strides = opts["strides"]
    except KeyError:
        raise ValueError(f"strides not set for {name}")
    logger.debug(f"strides set: {strides}")

    # TODO: perform check against strides for shape
    if conv_dim == 1:
        out = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=1,
            activation=actfn,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # TODO: correct, k_init_fn
            bias_initializer="zeros",  # TODO: implement
            kernel_regularizer=k_reg,
            bias_regularizer=b_reg,  # tf.zeros_initializer()
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name,
        )
    elif conv_dim == 2:
        out = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=(1, 1),  # TODO: implement
            activation=actfn,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # TODO: correct, k_init_fn
            bias_initializer="zeros",  # TODO: implement
            kernel_regularizer=k_reg,
            bias_regularizer=b_reg,  # tf.zeros_initializer()
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name,
        )
    elif conv_dim == 3:
        out = tf.keras.layers.Conv3D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=(1, 1, 1),
            activation=None,
            use_bias=True,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name,
        )
    else:
        raise ValueError(f"conv_dim {conv_dim} for layer {name} not in [1,2,3]")

    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))
    return out
