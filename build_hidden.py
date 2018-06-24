import tensorflow as tf
import numpy as np


def get_regulizer_fn(reg_str: str):

    # TODO: need to test/validate this contrib
    # TODO: need to allow modification for scale
    scale = 0.1

    if reg_str:
        reg_str = reg_str.lower()

    if reg_str == "":
        reg_fn = None  # default is glorot
    elif reg_str == "l1":
        reg_fn = tf.contrib.layers.l1_regularizer(scale, scope=None)
    elif reg_str == "l2":
        reg_fn = tf.contrib.layers.l2_regularizer(scale, scope=None)
    elif reg_str == "l1l2":
        # TODO: how/is this different from elastic nets
        reg_fn = tf.contrib.layers.l1_l2_regularizer(
            scale_l1=1.0, scale_l2=1.0, scope=None
        )
    else:
        # TODO: Error
        reg_fn = None

    return reg_fn


def get_initializer_fn(init_str: str):
    # NOTE: will use uniform (not normal) by default

    # elif opts["kernel_initializer"] == "he":
    # init_fn = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
    if init_str:
        init_str = init_str.lower()

    if init_str == "":
        init_fn = None  # default is glorot
    elif init_str == "glorot":
        init_fn = tf.glorot_uniform_initializer(seed=None, dtype=tf.float32)
    elif init_str == "zeros" or init_str == "zero":
        init_fn = tf.zeros_initializer(dtype=tf.float32)
    elif init_str == "ones" or init_str == "one":
        init_fn = tf.ones_initializer(dtype=tf.float32)
    elif init_str == "rand" or init_str == "random":
        # TODO: this will need a value for maxval
        init_fn = tf.random_uniform_initializer(
            minval=0, maxval=None, seed=None, dtype=tf.float32
        )
    elif init_str == "he":
        # TODO: unsure about this one...
        tf.contrib.layers.variance_scaling_initializer(
            factor=2.0, mode="FAN_IN", uniform=False, seed=None, dtype=tf.float32
        )
    else:
        # TODO: Error
        init_fn = None
    return init_fn


def get_activation_fn(act_str: str):

    if act_str:
        act_str = act_str.lower()

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
        # TODO: Error
        # realistically this should be caught by the initial check
        pass
    return act_fn


def build_conv2d_layer(cur_input, opts: dict, actfn, name: str):
    # TODO: default behavior is w/in the exception block, this may need to change
    # default is 3x3, stride = 1

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None

    try:
        k_reg = get_regulizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None

    try:
        b_reg = get_regulizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None

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

    return out


def build_dense_layer(cur_input, opts: dict, actfn, name: str):
    units = opts["units"]

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None

    try:
        k_reg = get_regulizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None

    try:
        b_reg = get_regulizer_fn(opts["bias_regularizer"])
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
