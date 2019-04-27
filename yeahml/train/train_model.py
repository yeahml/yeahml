import tensorflow as tf
import math
import numpy as np
import os
import sys
from typing import Any

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.build.load_params_onto_layer import init_params_from_file  # load params
from yeahml.build.components.loss import get_loss_fn
from yeahml.build.components.optimizer import get_optimizer


@tf.function
def train_step(model, x_batch, y_batch, loss_fn, optimizer, loss_avg, metrics):

    with tf.GradientTape() as tape:
        prediction = model(x_batch)

        # TODO: apply mask?

        loss = loss_fn(y_batch, prediction)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # TODO: convert to iterating list
    loss_avg(loss)

    # TODO: convert to iterating list
    metrics(prediction, y_batch)


@tf.function
def val_step(model, x_batch, y_batch, loss_fn, loss_avg, metrics):
    prediction = model(x_batch)
    loss = loss_fn(y_batch, prediction)
    loss_avg(loss)
    metrics(prediction, y_batch)


def train_model(model, MCd: dict, HCd: dict) -> dict:
    return_dict = {}

    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    # get model
    # get optimizer
    optimizer = get_optimizer(MCd)

    # get loss function
    loss_object = get_loss_fn(MCd["loss_fn"])

    # mean loss
    avg_train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)

    # get metrics
    # TODO: create list, will need to iterate through later
    train_metrics = tf.keras.metrics.MeanAbsoluteError(
        name="train_accuracy", dtype=tf.float32
    )
    val_metrics = tf.keras.metrics.MeanAbsoluteError(
        name="validation_accuracy", dtype=tf.float32
    )

    # get dataset
    tfr_train_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    train_ds = return_batched_iter("train", MCd, tfr_train_path)

    tfr_val_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    val_ds = return_batched_iter("train", MCd, tfr_val_path)

    best_val_loss = np.inf
    # TODO: train loop
    # TODO: loop metrics
    template_str: str = "epoch: {:3} train loss: {:.4f} | val loss: {:.4f}"
    for e in range(MCd["epochs"]):
        # TODO: abstract to fn to clear *all* metrics and loss objects
        avg_train_loss.reset_states()
        avg_val_loss.reset_states()
        train_metrics.reset_states()
        val_metrics.reset_states()

        # logger.debug("-> START iterating training dataset")
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            train_step(
                model,
                x_batch_train,
                y_batch_train,
                loss_object,
                optimizer,
                avg_train_loss,
                train_metrics,
            )
        # logger.debug("-> END iterating training dataset")

        # iterate validation after iterating entire training.. this will/should change
        # logger.debug("-> START iterating validation dataset")
        for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
            val_step(
                model, x_batch_val, y_batch_val, loss_object, avg_val_loss, val_metrics
            )

        # check save best metrics
        cur_val_loss = avg_val_loss.result()
        if cur_val_loss < best_val_loss:
            best_val_loss = cur_val_loss
            model.save_weights(os.path.join(MCd["save_weights_path"]))
            logger.debug("best params saved: val loss: {:.4f}".format(cur_val_loss))

        # logger.debug("-> END iterating validation dataset")

        # TODO: loop metrics
        logger.debug(
            template_str.format(e + 1, avg_train_loss.result(), avg_val_loss.result())
        )

    # return_dict = {}
    # loss history
    # return_dict["train_loss"] = best_train_loss
    # return_dict["val_loss"] = best_val_loss

    logger.info("[END] creating train_dict")
