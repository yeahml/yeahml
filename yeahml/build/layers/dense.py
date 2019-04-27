import tensorflow as tf
from yeahml.build.get_components import get_regularizer_fn, get_initializer_fn
from yeahml.build.components.activation import get_activation_fn
from yeahml.helper import fmt_tensor_info


def build_dense_layer(opts: dict, actfn, name: str, logger, g_logger):
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

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))

    ## add dropout, if indicated
    # try:
    #     dropout_rate = opts["dropout"]
    # except KeyError:
    #     dropout_rate = None
    # logger.debug("dropout_rate set: {}".format(dropout_rate))

    # if dropout_rate:
    #     out = tf.layers.dropout(
    #         inputs=out,
    #         rate=dropout_rate,
    #         noise_shape=None,
    #         seed=None,
    #         training=training,
    #         name=None,
    #     )
    #     logger.debug("tensor obj post dropout: {}".format(out))
    #     g_logger.info(">> dropout: {}".format(dropout_rate))

    logger.debug("[End] building: {}".format(name))
    return out
