import tensorflow as tf
import sys


def get_tf_dtype(dtype: str):
    # TODO: add type supports + error handling
    tf_dtype = None

    if dtype == "float32":
        tf_dtype = tf.float32
    elif dtype == "int64":
        tf_dtype = tf.int64
    elif dtype == "int32":
        tf_dtype = tf.int32
    elif dtype == "int8":
        tf_dtype = tf.int8
    elif dtype == "string":
        tf_dtype = tf.string
    else:
        sys.exit("Error: Exit: dtype {} not recognized/supported".format(dtype))

    return tf_dtype


def get_regularizer_fn(reg_str: str):

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


def get_optimizer(MCd: dict):
    opt = MCd["optimizer"].lower()
    optimizer = None
    if opt == "adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=MCd["lr"],
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name="Adam",
        )
    elif opt == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=MCd["lr"], name="GradientDescent"
        )
    elif opt == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=MCd["lr"],
            rho=0.95,
            epsilon=1e-08,
            use_locking=False,
            name="Adadelta",
        )
    elif opt == "adagrad":
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=MCd["lr"],
            initial_accumulator_value=0.1,
            use_locking=False,
            name="Adagrad",
        )
    # elif opt == "momentum":
    #     tf.train.MomentumOptimizer(
    #         learning_rate=MCd["lr"],
    #         momentum, # TODO: value
    #         use_locking=False,
    #         name="Momentum",
    #         use_nesterov=False,
    #     )
    elif opt == "ftrl":
        optimizer = tf.train.FtrlOptimizer(
            learning_rate=MCd["lr"],
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            use_locking=False,
            name="Ftrl",
            accum_name=None,
            linear_name=None,
            l2_shrinkage_regularization_strength=0.0,
        )
    elif opt == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=MCd["lr"],
            decay=0.9,
            momentum=0.0,
            epsilon=1e-10,
            use_locking=False,
            centered=False,
            name="RMSProp",
        )
    else:
        # TODO: error handle?
        # realistically this should be caught by the initial check
        pass

    return optimizer


def get_activation_fn(act_str: str):

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
    # elif act == "leaky":
    # act_fn = tf.nn.leay_relu
    elif act_str == "relu6":
        act_fn = tf.nn.relu6
    elif act_str == "identity":
        act_fn = tf.identity
    else:
        # TODO: Error logging
        # the reasoning here is that the relu is subjectively the most
        # common/default activation function in DNNs, but I don't LOVE this
        sys.exit("No activation function has been set")

    return act_fn


def get_logits_and_preds(loss_str: str, hidden_out, num_classes: int, logger) -> tuple:
    # create the output layer (logits and preds) based on the type of loss function used.
    if loss_str == "sigmoid":
        logits = tf.layers.dense(hidden_out, num_classes, name="logits")
        preds = tf.sigmoid(logits, name="y_proba")
    elif loss_str == "softmax":
        logits = tf.layers.dense(hidden_out, num_classes, name="logits")
        preds = tf.nn.softmax(logits, name="y_proba")
    elif loss_str == "softmax_binary_segmentation_temp":
        logits = hidden_out
        preds = tf.nn.softmax(logits, name="y_proba")
    elif loss_str == "mse" or loss_str == "rmse":
        logits = tf.layers.dense(hidden_out, num_classes, name="logits")
        preds = logits
    else:
        logger.fatal("preds cannot be created as: {}".format(loss_str))
        sys.exit("final_type: {} -- is not supported or defined.".format(loss_str))
    logger.debug("pred created as {}: {}".format(loss_str, preds))

    return (logits, preds)


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


def get_run_options(temp_trace_level: str):

    if temp_trace_level == "full":
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    elif temp_trace_level == "software":
        run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
    elif temp_trace_level == "hardware":
        run_options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
    elif temp_trace_level == "None":
        run_options = None
    else:
        run_options = None

    return run_options
