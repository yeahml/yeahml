import tensorflow as tf
from yeahml.helper import fmt_tensor_info


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

    logger.debug("tensor ob embedding_lookup: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))

    # TODO: VIZUALIZING EMBEDDINGS - https://www.tensorflow.org/guide/embedding

    return out


def build_batch_normalization_layer(
    cur_input, training, opts: dict, name: str, logger, g_logger
):
    # TODO: no opts yet.
    out = tf.contrib.layers.batch_norm(
        cur_input,
        decay=0.999,
        center=True,
        scale=False,
        epsilon=0.001,
        activation_fn=None,
        param_initializers=None,
        param_regularizers=None,
        updates_collections=None,
        is_training=training,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        batch_weights=None,
        fused=None,
        zero_debias_moving_mean=False,
        scope=None,
        renorm=False,
        renorm_clipping=None,
        renorm_decay=0.99,
        adjustment=None,
    )
    # out = tf.layers.batch_normalization(
    #     cur_input,
    #     axis=-1,
    #     momentum=0.99,
    #     epsilon=0.001,
    #     center=True,
    #     scale=True,
    #     beta_initializer=tf.zeros_initializer(),
    #     gamma_initializer=tf.ones_initializer(),
    #     moving_mean_initializer=tf.zeros_initializer(),
    #     moving_variance_initializer=tf.ones_initializer(),
    #     beta_regularizer=None,
    #     gamma_regularizer=None,
    #     beta_constraint=None,
    #     gamma_constraint=None,
    #     training=training,
    #     trainable=True,
    #     name=None,
    #     reuse=None,
    #     renorm=False,
    #     renorm_clipping=None,
    #     renorm_momentum=0.99,
    #     fused=None,
    #     virtual_batch_size=None,
    #     adjustment=None,
    # )

    logger.debug("tensor ob batch_norm: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))

    return out

