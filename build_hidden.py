import tensorflow as tf
import numpy as np

from helper import print_tensor_info
from get_components import get_regularizer_fn, get_initializer_fn, get_activation_fn


def build_conv2d_layer(cur_input, opts: dict, actfn, name: str, G_PRINT: bool):
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None

    filters = opts["filters"]
    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        kernel_size = 3

    try:
        # TODO: create func (w/error handling) for this
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 1

    if not name:
        name = "unnamed_conv_layer"

    # trainable = opts["trainable"]
    trainable = True

    out = tf.layers.conv2d(
        cur_input,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=actfn,
        kernel_initializer=k_init_fn,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        trainable=trainable,
        name=name,
    )

    if G_PRINT:
        print_tensor_info(out)

    return out


def build_dense_layer(cur_input, training, opts: dict, actfn, name: str, G_PRINT: bool):
    units = opts["units"]

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None

    # trainable = opts["trainable"]
    trainable = True
    out = tf.layers.dense(
        inputs=cur_input,
        units=units,
        activation=actfn,
        kernel_initializer=k_init_fn,
        kernel_regularizer=k_reg,
        bias_regularizer=b_reg,
        trainable=trainable,
        name=name,
    )

    if G_PRINT:
        print_tensor_info(out)

    ## add dropout
    # this block isn't very elegant...
    try:
        dropout_rate = opts["dropout"]
    except KeyError:
        dropout_rate = None

    if dropout_rate:
        # apply dropout
        out = tf.layers.dropout(
            inputs=out,
            rate=dropout_rate,
            noise_shape=None,
            seed=None,
            training=training,
            name=None,
        )

        if G_PRINT:
            print(">> dropout: {}".format(dropout_rate))

    return out


def build_pool_layer(cur_input, training, opts: dict, name: str, G_PRINT: bool):

    try:
        if opts:
            pool_size = opts["pool_size"]
        else:
            pool_size = [2, 2]
    except KeyError:
        pool_size = [2, 2]

    try:
        if opts:
            strides = opts["strides"]
        else:
            strides = 2
    except KeyError:
        strides = 2

    if not name:
        name = "unnamed_pool2d_layer"

    out = tf.layers.max_pooling2d(
        cur_input, pool_size=pool_size, strides=strides, name=name
    )

    if G_PRINT:
        print_tensor_info(out)

    ## add dropout
    # this block isn't very elegant...
    try:
        if opts:
            dropout_rate = opts["dropout"]
        else:
            dropout_rate = None
    except KeyError:
        dropout_rate = None

    if dropout_rate:
        # apply dropout
        out = tf.layers.dropout(
            inputs=out,
            rate=dropout_rate,
            noise_shape=None,
            seed=None,
            training=training,
            name=None,
        )
        if G_PRINT:
            print(">> dropout: {}".format(dropout_rate))

    return out


def build_hidden_block(X, training, MCd: dict, ACd: dict):
    # in: X
    # out: last layer before logits

    # cur_input will be updated each iteration such that on
    # the next iteration it will be the input to the following layer
    cur_input = X

    # G_PRINT is used as bool to determine whether information
    # about the graph should be printed
    try:
        G_PRINT = MCd["print_g_spec"]
    except:
        G_PRINT = False
        pass

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
            try:
                actfn = get_activation_fn(MCd["def_act"])
            except KeyError:
                # the reasoning here is that the relu is subjectively the most
                # common/default activation function in DNNs, but I don't LOVE this
                actfn = get_activation_fn("relu")

        ltype = layer_info["type"].lower()
        if ltype == "conv2d":
            cur_input = build_conv2d_layer(cur_input, opts, actfn, l_name, G_PRINT)
        elif ltype == "dense":
            # ---- this block is 'dumb' but works --------------
            # this is necessary because if the last block was a pool
            # or conv, we need to flatten layer before we add a dense layer
            prev_ltype_key = list(ACd["layers"])[i - 1]
            prev_ltype = ACd["layers"][prev_ltype_key]["type"]
            if prev_ltype == "conv2d" or prev_ltype == "pooling2d":
                # flatten
                last_shape = int(np.prod(cur_input.get_shape()[1:]))
                cur_input = tf.reshape(cur_input, shape=[-1, last_shape])
                if G_PRINT:
                    print(">> flatten: {}".format(cur_input.shape))
            # --------------------------------------------------
            cur_input = build_dense_layer(
                cur_input, training, opts, actfn, l_name, G_PRINT
            )
        elif ltype == "pooling2d":
            cur_input = build_pool_layer(cur_input, training, opts, l_name, G_PRINT)
        else:
            print("ruh roh.. this is currently a fatal err")

    return cur_input
