import tensorflow as tf
from yeahml.build.get_components import get_initializer_fn
from yeahml.build.components.regularizer import get_regularizer_fn
from yeahml.build.components.activation import get_activation_fn
from typing import Any

from yeahml.helper import fmt_tensor_info


def build_deconv_layer(opts: dict, actfn, name: str, logger, g_logger) -> Any:
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None
    logger.debug(f"k_init_fn set: {k_init_fn}")

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None
    logger.debug(f"k_reg set: {k_reg}")

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None
    logger.debug(f"b_reg set: {b_reg}")

    try:
        data_format = opts["data_format"]
    except:
        data_format = None
    logger.debug(f"data_format set: {data_format}")

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
    logger.debug(f"padding set: {padding}")

    if not name:
        name = "unnamed_conv_layer"
    logger.debug(f"name set: {name}")

    try:
        conv_dim = opts["dimension"]  # TOOD: shorthand 'dim'
    except KeyError:
        raise ValueError(f"convolution dim not specified for {name}")

    # TODO: make sure these correspond to the current layer type
    try:
        filters = opts["filters"]
    except KeyError:
        raise ValueError(f"filters not set for {name}")
    logger.debug(f"filters set: {filters}")

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
    if conv_dim == 2:
        out = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,  # (1,1)
            padding=padding,
            output_padding=None,
            data_format=None,
            dilation_rate=(1, 1),
            activation=actfn,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # TODO: correct, k_init_fn
            bias_initializer="zeros",  # TODO: implement
            kernel_regularizer=k_reg,
            bias_regularizer=b_reg,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name,
        )
    elif conv_dim == 3:
        out = tf.keras.layers.Conv3DTranspose(
            filters,
            kernel_size,
            strides=strides,  # (1,1,1)
            padding=padding,
            output_padding=None,
            data_format=None,
            activation=actfn,
            use_bias=True,
            kernel_initializer="glorot_uniform",  # TODO: correct, k_init_fn
            bias_initializer="zeros",  # TODO: implement
            kernel_regularizer=k_reg,
            bias_regularizer=b_reg,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            name=name,
        )
    else:
        raise ValueError(f"conv_dim {conv_dim} for layer {name} not in [2,3]")

    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")
    return out
