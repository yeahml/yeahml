import tensorflow as tf

from yeahml.helper import fmt_tensor_info


def build_embedding_layer(opts: dict, name: str, logger, g_logger):
    if not name:
        name = "unnamed_embedding_layer"
    logger.debug(f"name set: {name}")

    # TODO: the default vocabulary_size size does not make sense here
    try:
        if opts:
            input_dim = opts["vocabulary_size"]
        else:
            input_dim = 100
    except KeyError:
        input_dim = 100
    logger.debug(f"input_dim (vocabulary size) set: {input_dim}")

    # TODO: the default embedding_size size does not make sense here
    try:
        if opts:
            output_dim = opts["embedding_size"]
        else:
            output_dim = 2
    except KeyError:
        output_dim = 2
    logger.debug(f"output_dim (embedding size) set: {output_dim}")

    out = tf.keras.layers.Embedding(
        input_dim=input_dim,
        output_dim=output_dim,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        name=name,
    )

    logger.debug(f"tensor ob embedding_lookup: {out}")
    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")

    # TODO: VIZUALIZING EMBEDDINGS - https://www.tensorflow.org/guide/embedding

    return out


def build_batch_normalization_layer(opts: dict, name: str, logger, g_logger):

    out = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        trainable=True,
        virtual_batch_size=None,
        adjustment=None,
        name=name,
    )

    logger.debug(f"tensor ob batch_norm: {name}")
    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")

    return out


def build_flatten_layer(opts: dict, name: str, logger, g_logger):

    out = tf.keras.layers.Flatten()

    logger.debug(f"tensor ob batch_norm: {name}")
    g_logger.info(f"{fmt_tensor_info(out)}")
    logger.debug(f"[End] building: {name}")

    return out
