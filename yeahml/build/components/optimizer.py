import tensorflow as tf
from typing import Any


def get_optimizer(MCd: dict) -> Any:

    # TODO: this functionality should mimic that of Layers

    # TODO: this should be created in the config
    opt = MCd["optimizer"].lower()
    # TODO: this check should be pushed to the config logic
    if opt:
        opt = opt.lower()
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
        # realistically this should be caught by the initial check
        raise NotImplementedError

    return optimizer
