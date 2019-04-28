import tensorflow as tf
from typing import Any
from yeahml.helper import fmt_tensor_info


def build_dropout_layer(opts, l_name, logger, g_logger) -> Any:
    try:
        dropout_type = opts["type"]
    except KeyError:
        # default to 'standard' dropout
        dropout_type = None

    try:
        rate = opts["rate"]
    except KeyError:
        raise ValueError(f"No dropout rate specified for out: {l_name}")

    if not dropout_type:
        out = tf.keras.layers.Dropout(rate, noise_shape=None, seed=None)
    elif dropout_type == "alpha":
        out = tf.keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
    elif dropout_type == "gaussian":
        out = tf.keras.layers.GaussianDropout(rate)
    elif dropout_type == "spatial":
        try:
            spatial = opts["type"]
        except KeyError:
            raise ValueError(f"No spatial dropout dimmension specified: {spatial}")

        if spatial == 1:
            out = tf.keras.layers.SpatialDropout1D(rate)
        elif spatial == 2:
            out = tf.keras.layers.SpatialDropout2D(rate, data_format=None)
        elif spatial == 3:
            out = tf.keras.layers.SpatialDropout3D(rate, data_format=None)
        else:
            raise ValueError(f"spatial dimension: {spatial} not in set([1,2,3])")

    else:
        raise NotImplementedError(f"dropout type {dropout_type} is not implemented yet")

    g_logger.info(f"{fmt_tensor_info(out)}")

    logger.debug(f"[End] building: {l_name}")
    return out

