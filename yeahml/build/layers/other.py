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

    logger.debug("tensor obj pre dropout: {}".format(out))
    g_logger.info("{}".format(fmt_tensor_info(out)))
    logger.debug("[End] building: {}".format(name))

    # TODO: VIZUALIZING EMBEDDINGS - https://www.tensorflow.org/guide/embedding

    return out
