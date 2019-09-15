from typing import Any

import tensorflow as tf


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
        raise ValueError(f"Error: Exit: dtype {dtype} not recognized/supported")

    return tf_dtype


def get_lr_schedule():
    # TODO: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/optimizers/schedules/ExponentialDecay
    raise NotImplementedError


# def get_logits_and_preds(loss_str: str, hidden_out, num_classes: int, logger) -> tuple:
#     # create the output layer (logits and preds) based on the type of loss function used.
#     raise NotImplementedError
#     logger.debug("pred created as {}: {}".format(loss_str, preds))

#     return (logits, preds)
