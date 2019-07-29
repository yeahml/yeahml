import tensorflow as tf
import sys

from typing import Any


def get_tf_dtype(dtype: str):
    # TODO: add type supports + error handling
    tf_dtype = None

    # floats
    if dtype == "float16":
        tf_dtype = tf.float16
    elif dtype == "float32":
        tf_dtype = tf.float32
    elif dtype == "float64":
        tf_dtype = tf.float64
    # ints
    elif dtype == "int8":
        tf_dtype = tf.int8
    elif dtype == "int16":
        tf_dtype = tf.int16
    elif dtype == "int32":
        tf_dtype = tf.int32
    elif dtype == "int64":
        tf_dtype = tf.int64

    # other
    elif dtype == "string":
        tf_dtype = tf.string
    else:
        sys.exit("Error: Exit: dtype {} not recognized/supported".format(dtype))

    return tf_dtype


def get_lr_schedule():
    # TODO: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/optimizers/schedules/ExponentialDecay
    raise NotImplementedError


# def get_logits_and_preds(loss_str: str, hidden_out, num_classes: int, logger) -> tuple:
#     # create the output layer (logits and preds) based on the type of loss function used.
#     raise NotImplementedError
#     logger.debug("pred created as {}: {}".format(loss_str, preds))

#     return (logits, preds)


def get_initializer_fn(init_str: str):
    # NOTE: will use uniform (not normal) by default

    if init_str:
        init_str = init_str.lower()
    if init_str == "":
        init_fn = None  # default is glorot
    elif init_str == "glorot":
        raise NotImplementedError
    elif init_str == "zeros" or init_str == "zero":
        init_fn = tf.zeros_initializer(dtype=tf.float32)
    elif init_str == "ones" or init_str == "one":
        init_fn = tf.ones_initializer(dtype=tf.float32)
    elif init_str == "rand" or init_str == "random":
        # TODO: this will need a value for maxval
        raise NotImplementedError
    elif init_str == "he":
        raise NotImplementedError
    else:
        # TODO: Error
        init_fn = None
    return init_fn

