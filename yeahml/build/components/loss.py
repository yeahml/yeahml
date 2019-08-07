import tensorflow as tf
from typing import Any
from tensorflow.python.keras.utils import losses_utils  # ugly
import inspect
from yeahml.build.components.configure import copy_func


def return_available_losses():

    LOSS_FUNCTIONS = {}
    available_keras_losses = tf.losses.__dict__

    for opt_name, opt_func in available_keras_losses.items():
        if callable(opt_func) and not inspect.isclass(opt_func):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                LOSS_FUNCTIONS[opt_name.lower()] = {}
                LOSS_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = list(opt_func.__code__.co_varnames)
                args = [a for a in args if a not in ["y_pred", "y_true"]]
                LOSS_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return LOSS_FUNCTIONS


def return_loss(loss_str):
    avail_losses = return_available_losses()
    try:
        loss = avail_losses[loss_str]
    except KeyError:
        raise KeyError(
            f"activation {loss_str} not available in options {avail_losses.keys()}"
        )

    return loss


def configure_loss(opt_dict):

    try:
        cur_type = opt_dict["type"]
    except TypeError:
        # TODO: could include more helpful message here if the type is an initializer option
        raise TypeError(
            f"config for initialier does not specify a 'type'. Current specified options:({opt_dict})."
        )
    loss_obj = return_loss(cur_type.lower())
    loss_fn = loss_obj["function"]

    try:
        options = opt_dict["options"]
    except KeyError:
        options = None

    if options:
        if not set(opt_dict["options"].keys()).issubset(loss_obj["func_args"]):
            raise ValueError(
                f"options {opt_dict['options'].keys()} not in {init_obj['func_args']}"
            )
        loss_fn = copy_func(loss_fn)
        var_list = list(loss_fn.__code__.co_varnames)
        # TODO: there must be a more `automatic` way to filter these
        var_list = [
            e
            for e in var_list
            if e not in ("y_pred", "y_true") and not e.startswith("_")
        ]
        cur_defaults_list = list(loss_fn.__defaults__)
        for ao, v in options.items():
            arg_index = var_list.index(ao)
            # TODO: same type assertion?
            cur_defaults_list[arg_index] = v
        loss_fn.__defaults__ = tuple(cur_defaults_list)

    return loss_fn


# def get_loss_fn(loss_str: str) -> Any:

#     # TODO: this functionality should mimic that of Layers

#     # TODO: this check should be pushed to the config logic
#     if loss_str:
#         loss_str = loss_str.lower()

#     loss_obj = None
#     if loss_str == "binarycrossentropy":
#         loss_obj = tf.losses.BinaryCrossentropy(
#             from_logits=False,
#             label_smoothing=0,
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="binary_crossentropy",
#         )
#     elif loss_str == "categoricalcrossentropy":
#         loss_obj = tf.losses.CategoricalCrossentropy(
#             from_logits=False,
#             label_smoothing=0,
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="categorical_crossentropy",
#         )
#     elif loss_str == "categoricalhinge":
#         loss_obj = tf.losses.CategoricalHinge(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="categorical_hinge",
#         )
#     elif loss_str == "cosinesimilarity":
#         # TODO: specify axis
#         loss_obj = tf.losses.CosineSimilarity(
#             axis=-1,
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="cosine_similarity",
#         )
#     elif loss_str == "hinge":
#         loss_obj = tf.losses.Hinge(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name=None
#         )
#     elif loss_str == "huber":
#         loss_obj = tf.losses.Huber(
#             delta=1.0,
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="huber_loss",
#         )
#     elif loss_str == "kldivergence":
#         loss_obj = tf.losses.KLDivergence(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="kullback_leibler_divergence",
#         )
#     elif loss_str == "logcosh":
#         loss_obj = tf.losses.LogCosh(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="logcosh"
#         )
#     elif loss_str == "logloss":
#         loss_obj = tf.losses.LogLoss(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="logloss"
#         )
#     # elif loss_str == "loss":
#     #     loss_obj = tf.losses.Loss() # loss base class, for creating own
#     # class MeanSquaredError(Loss):
#     # def call(self, y_true, y_pred):
#     #     y_pred = ops.convert_to_tensor(y_pred)
#     #     y_true = math_ops.cast(y_true, y_pred.dtype)
#     #     return K.mean(math_ops.square(y_pred - y_true), axis=-1)
#     elif loss_str == "meanabsoluteerror":
#         loss_obj = tf.losses.MeanAbsoluteError(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="mean_absolute_error",
#         )
#     elif loss_str == "meanabsolutepercentageerror":
#         loss_obj = tf.losses.MeanAbsolutePercentageError(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="mean_absolute_percentage_error",
#         )
#     elif loss_str == "meansquarederror":
#         loss_obj = tf.losses.MeanSquaredError(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="mean_squared_error",
#         )
#     elif loss_str == "meansquaredlogarithmicerror":
#         loss_obj = tf.losses.MeanSquaredLogarithmicError(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name="mean_squared_logarithmic_error",
#         )
#     elif loss_str == "poisson":
#         # loss = y_pred - y_true * log(y_pred)
#         loss_obj = tf.losses.Poisson(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="poisson"
#         )
#     elif loss_str == "sparsecategoricalcrossentropy":
#         loss_obj = tf.losses.SparseCategoricalCrossentropy(
#             from_logits=False,
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
#             name=None,
#         )
#     elif loss_str == "squaredhinge":
#         loss_obj = tf.losses.SquaredHinge(
#             reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="squared_hinge"
#         )
#     else:
#         raise NotImplementedError

#     return loss_obj
