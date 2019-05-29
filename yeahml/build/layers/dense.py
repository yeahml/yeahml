import tensorflow as tf
from yeahml.build.get_components import get_initializer_fn
from yeahml.build.components.regularizer import get_regularizer_fn
from yeahml.build.components.activation import get_activation_fn
from yeahml.helper import fmt_tensor_info


def build_dense_layer(opts: dict, actfn, name: str, logger, g_logger):
    try:
        units = opts["units"]
    except KeyError:
        raise ValueError(f"no units specified for {name}")
    logger.debug(f"units set: {units}")

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

    out = tf.keras.layers.Dense(
        units,
        activation=actfn,
        use_bias=True,
        kernel_initializer="glorot_uniform",  # k_init_fn
        bias_initializer="zeros",
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=name,
    )

    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")
    return out
