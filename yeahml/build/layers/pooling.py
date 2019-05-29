import tensorflow as tf
import sys
from yeahml.helper import fmt_tensor_info
from typing import Any


def build_pooling_layer(opts: dict, name: str, logger: Any, g_logger: Any) -> Any:
    if not name:
        name = "unnamed_pooling_layer"
    logger.debug(f"name set: {name}")

    # 1,2,3D
    try:
        pool_dim = opts["dimension"]  # TOOD: shorthand 'dim'
    except KeyError:
        raise ValueError(f"pooling dim not specified for {name}")
    logger.debug(f"pool_dim set: {pool_dim}")

    # avg, max
    try:
        pool_type = opts["avg_max"]
    except KeyError:
        raise ValueError(f"pooling type not specified for {name}")
    logger.debug(f"pool_type set: {pool_type}")

    # global/not
    try:
        pool_global = opts["global"]
    except KeyError:
        pool_global = False
    logger.debug(f"pool_global set: {pool_global}")

    # layer information
    try:
        pool_size = opts["pool_size"]
    except KeyError:
        # TODO: ensure the value is "reasonable"
        raise ValueError(f"pool_size not specified for {name}")
    logger.debug(f"pool_size set: {pool_size}")

    try:
        strides = opts["strides"]
    except KeyError:
        # TODO: ensure the value is "reasonable"
        raise ValueError(f"strides not specified for {name}")
    logger.debug(f"strides set: {strides}")

    try:
        padding = opts["padding"]
    except KeyError:
        # TODO: this is opposite the default on tf
        padding = "same"
    logger.debug(f"padding set: {padding}")

    try:
        data_format = opts["data_format"]
    except KeyError:
        data_format = "channels_last"
    logger.debug(f"data_format set: {data_format}")

    if pool_type == "avg":
        if pool_global:
            if pool_dim == 1:
                out = tf.keras.layers.GlobalAveragePooling1D(
                    data_format=data_format, name=name
                )
            elif pool_dim == 2:
                out = tf.keras.layers.GlobalAveragePooling2D(
                    data_format=data_format, name=name
                )
            elif pool_dim == 3:
                out = tf.keras.layers.GlobalAveragePooling3D(
                    data_format=data_format, name=name
                )
            else:
                raise NotImplementedError(
                    f"pool_dim {pool_dim} not implemented. only [1,2,3]"
                )
        else:
            if pool_dim == 1:
                out = tf.keras.layers.AveragePooling1D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            elif pool_dim == 2:
                out = tf.keras.layers.AveragePooling2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            elif pool_dim == 3:
                out = tf.keras.layers.AveragePooling3D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            else:
                raise NotImplementedError(
                    f"pool_dim {pool_dim} not implemented. only [1,2,3]"
                )
    elif pool_type == "max":
        if pool_global:
            if pool_dim == 1:
                out = tf.keras.layers.GlobalMaxPool1D(
                    data_format=data_format, name=name
                )
            elif pool_dim == 2:
                out = tf.keras.layers.GlobalMaxPool2D(
                    data_format=data_format, name=name
                )
            elif pool_dim == 3:
                out = tf.keras.layers.GlobalMaxPool3D(
                    data_format=data_format, name=name
                )
            else:
                raise NotImplementedError(
                    f"pool_dim {pool_dim} not implemented. only [1,2,3]"
                )
        else:
            if pool_dim == 1:
                out = tf.keras.layers.MaxPool1D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            elif pool_dim == 2:
                out = tf.keras.layers.MaxPool2D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            elif pool_dim == 3:
                out = tf.keras.layers.MaxPool3D(
                    pool_size=pool_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    name=name,
                )
            else:
                raise NotImplementedError(
                    f"pool_dim {pool_dim} not implemented. only [1,2,3]"
                )
    else:
        raise NotImplementedError(
            f"pool_type {pool_type} not implemented. only [avg,max]"
        )

    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")
    return out
