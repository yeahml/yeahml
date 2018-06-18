import tensorflow as tf
import numpy as np


def get_activation_fn(act_str: str):

    act_fn = None
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
        act_gn = tf.nn.relu
    # elif act == "leaky":
    # act_fn = tf.nn.leay_relu
    elif act_str == "relu6":
        act_fn = tf.nn.relu6
    else:
        # TODO: error handle?
        # realistically this should be caught by the initial check
        pass
    return act_fn


def build_conv2d_layer(cur_input, opts: dict, actfn, name: str):
    # TODO: convert to lower API
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    filters = opts["filters"]
    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        kernel_size = 3

    try:
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 1

    if not name:
        name = "unnamed_conv_layer"

    out = tf.layers.conv2d(
        cur_input,
        filters=filters,
        kernel_size=kernel_size,
        activation=actfn,
        padding=padding,
        strides=strides,
        name=name,
    )

    return out


def build_dense_layer(cur_input, opts: dict, actfn, name: str):
    # TODO: convert to lower API
    units = opts["units"]
    out = tf.layers.dense(cur_input, units, activation=actfn, name=name)
    return out


def build_pool_layer(cur_input, opts: dict, name: str):

    try:
        pool_size = opts["pool_size"]
    except KeyError:
        pool_size = [2, 2]

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 2

    if not name:
        name = "unnamed_pool2d_layer"

    out = tf.layers.max_pooling2d(
        cur_input, pool_size=pool_size, strides=strides, name=name
    )

    return out


def build_hidden_block(X, MCd: dict, ACd: dict):
    # in: X
    # out: last layer before logits

    # cur_input will be updated each iteration such that on
    # the next iteration it will be the input to the following layer
    cur_input = X

    with tf.name_scope("hidden"):
        # build each layer based on the (ordered) yaml specification
        for i, l_name in enumerate(ACd["layers"]):
            layer_info = ACd["layers"][str(l_name)]

            try:
                opts = layer_info["options"]
            except:
                opts = None
                pass

            try:
                actfn_str = layer_info["activation"]
                actfn = get_activation_fn(actfn_str)
            except KeyError:
                # TODO: this seems dumb..
                actfn = get_activation_fn(MCd["def_act"])
                pass

            ltype = layer_info["type"].lower()
            if ltype == "conv2d":
                cur_input = build_conv2d_layer(cur_input, opts, actfn, l_name)
            elif ltype == "dense":
                # ---- this block is 'dumb' but works --------------
                prev_ltype_key = list(ACd["layers"])[i - 1]
                prev_ltype = ACd["layers"][prev_ltype_key]["type"]
                # --------------------------------------------------
                if prev_ltype == "conv2d" or prev_ltype == "pooling2d":
                    # flatten
                    last_shape = int(np.prod(cur_input.get_shape()[1:]))
                    cur_input = tf.reshape(cur_input, shape=[-1, last_shape])
                cur_input = build_dense_layer(cur_input, opts, actfn, l_name)
            elif ltype == "pooling2d":
                cur_input = build_pool_layer(cur_input, opts, l_name)
            else:
                print("ruh roh.. this is currently a fatal err")

    return cur_input
