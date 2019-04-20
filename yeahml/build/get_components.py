import tensorflow as tf
import sys


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


def get_regularizer_fn(reg_str: str):

    # TODO: need to allow modification for scale
    scale = 0.1

    if reg_str:
        reg_str = reg_str.lower()

    if reg_str == "":
        reg_fn = None  # default is glorot
    elif reg_str == "l1":
        raise NotImplementedError
    elif reg_str == "l2":
        raise NotImplementedError
    elif reg_str == "l1l2":
        raise NotImplementedError
    else:
        # TODO: Error
        reg_fn = None

    return reg_fn


def get_optimizer_schedule():
    # TODO: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/optimizers/schedules/ExponentialDecay
    raise NotImplementedError


def get_optimizer(MCd: dict):
    opt = MCd["optimizer"].lower()
    optimizer = None
    # learning_rate=
    if opt == "adadelta":
        optimizer = tf.optimizers.Adadelta(
            learning_rate=MCd["lr"], rho=0.95, epsilon=1e-07, name="Adadelta"
        )
    elif opt == "adagrad":
        optimizer = tf.optimizers.Adagrad(
            learning_rate=MCd["lr"],
            initial_accumulator_value=0.1,
            epsilon=1e-07,
            name="Adagrad",
        )
    elif opt == "adam":
        optimizer = tf.optimizers.Adam(
            learning_rate=MCd["lr"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )
    elif opt == "adamax":
        optimizer = tf.optimizers.Adamax(
            learning_rate=MCd["lr"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            name="Adamax",
        )
    elif opt == "ftrl":
        optimizer = tf.optimizers.Ftrl(
            learning_rate=MCd["lr"],
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            name="Ftrl",
            l2_shrinkage_regularization_strength=0.0,
        )
    elif opt == "nadam":
        optimizer = tf.optimizers.Nadam(
            learning_rate=MCd["lr"],
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            name="Nadam",
        )
    elif opt == "rmsprop":
        optimizer = tf.optimizers.RMSprop(
            learning_rate=MCd["lr"],
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name="RMSprop",
        )
    elif opt == "sgd":
        optimizer = tf.optimizers.SGD(
            learning_rate=MCd["lr"], momentum=0.0, nesterov=False, name="SGD"
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
        sys.exit("No activation function has been set")

    return act_fn


def get_logits_and_preds(loss_str: str, hidden_out, num_classes: int, logger) -> tuple:
    # create the output layer (logits and preds) based on the type of loss function used.
    raise NotImplementedError
    logger.debug("pred created as {}: {}".format(loss_str, preds))

    return (logits, preds)


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

