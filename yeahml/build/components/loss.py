import tensorflow as tf
from typing import Any
from tensorflow.python.keras.utils import losses_utils  # ugly


def get_loss_fn(loss_str: str) -> Any:

    # TODO: this functionality should mimic that of Layers

    # TODO: this check should be pushed to the config logic
    if loss_str:
        loss_str = loss_str.lower()

    loss_obj = None
    if loss_str == "binarycrossentropy":
        loss_obj = tf.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="binary_crossentropy",
        )
    elif loss_str == "categoricalcrossentropy":
        loss_obj = tf.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="categorical_crossentropy",
        )
    elif loss_str == "categoricalhinge":
        loss_obj = tf.losses.CategoricalHinge(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="categorical_hinge",
        )
    elif loss_str == "cosinesimilarity":
        # TODO: specify axis
        loss_obj = tf.losses.CosineSimilarity(
            axis=-1,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="cosine_similarity",
        )
    elif loss_str == "hinge":
        loss_obj = tf.losses.Hinge(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name=None
        )
    elif loss_str == "huber":
        loss_obj = tf.losses.Huber(
            delta=1.0,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="huber_loss",
        )
    elif loss_str == "kldivergence":
        loss_obj = tf.losses.KLDivergence(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="kullback_leibler_divergence",
        )
    elif loss_str == "logcosh":
        loss_obj = tf.losses.LogCosh(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="logcosh"
        )
    elif loss_str == "logloss":
        loss_obj = tf.losses.LogLoss(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="logloss"
        )
    # elif loss_str == "loss":
    #     loss_obj = tf.losses.Loss() # loss base class, for creating own
    # class MeanSquaredError(Loss):
    # def call(self, y_true, y_pred):
    #     y_pred = ops.convert_to_tensor(y_pred)
    #     y_true = math_ops.cast(y_true, y_pred.dtype)
    #     return K.mean(math_ops.square(y_pred - y_true), axis=-1)
    elif loss_str == "meanabsoluteerror":
        loss_obj = tf.losses.MeanAbsoluteError(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="mean_absolute_error",
        )
    elif loss_str == "meanabsolutepercentageerror":
        loss_obj = tf.losses.MeanAbsolutePercentageError(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="mean_absolute_percentage_error",
        )
    elif loss_str == "meansquarederror":
        loss_obj = tf.losses.MeanSquaredError(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="mean_squared_error",
        )
    elif loss_str == "meansquaredlogarithmicerror":
        loss_obj = tf.losses.MeanSquaredLogarithmicError(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name="mean_squared_logarithmic_error",
        )
    elif loss_str == "poisson":
        # loss = y_pred - y_true * log(y_pred)
        loss_obj = tf.losses.Poisson(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="poisson"
        )
    elif loss_str == "sparsecategoricalcrossentropy":
        loss_obj = tf.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
            name=None,
        )
    elif loss_str == "squaredhinge":
        loss_obj = tf.losses.SquaredHinge(
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE, name="squared_hinge"
        )
    else:
        raise NotImplementedError

    return loss_obj
