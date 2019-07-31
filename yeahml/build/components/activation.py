import tensorflow as tf
from typing import Any


def get_activation_fn(act_str: str) -> Any:

    # TODO: this functionality should mimic that of Layers

    # TODO: this check should be pushed to the config logic
    if act_str:
        act_str = act_str.lower()

    act_fn = None  # TODO: this should maybe be an identity function
    if act_str == "sigmoid":
        act_fn = tf.sigmoid
    elif act_str == "tanh":
        act_fn = tf.tanh
    elif act_str == "elu":
        act_fn = tf.nn.elu
    elif act_str == "selu":
        act_fn = tf.nn.selu
    elif act_str == "softplus":
        act_fn = tf.nn.softplus
    elif act_str == "softsign":
        act_fn = tf.nn.softsign
    elif act_str == "relu":
        act_fn = tf.nn.relu
    elif act_str == "leaky":
        act_fn = tf.nn.leaky_relu
    elif act_str == "relu6":
        act_fn = tf.nn.relu6
    elif act_str == "identity":
        act_fn = tf.identity
    else:
        # TODO: Error logging
        # the reasoning here is that the relu is subjectively the most
        # common/default activation function in DNNs, but I don't LOVE this
        raise NotImplementedError

    return act_fn
