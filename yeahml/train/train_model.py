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
from yeahml.build.components.metrics import get_metrics_fn


@tf.function
def train_step(model, x_batch, y_batch, loss_fn, optimizer, loss_avg, metrics):

    with tf.GradientTape() as tape:
        prediction = model(x_batch)

        # TODO: apply mask?

        loss = loss_fn(y_batch, prediction)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # NOTE: only allow one loss
    loss_avg(loss)

    # TODO: ensure pred, gt order
    for train_metric in metrics:
        train_metric(prediction, y_batch)


@tf.function
def val_step(model, x_batch, y_batch, loss_fn, loss_avg, metrics):
    prediction = model(x_batch)
    loss = loss_fn(y_batch, prediction)

    # NOTE: only allow one loss
    loss_avg(loss)

    # TODO: ensure pred, gt order
    for val_metric in metrics:
        val_metric(prediction, y_batch)


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
    train_metric_fns = []
    val_metric_fns = []
    metric_order = []
    for metric in MCd["met_set"]:
        train_metric = get_metrics_fn(metric)
        train_metric_fns.append(train_metric)
        val_metric = get_metrics_fn(metric)
        val_metric_fns.append(val_metric)
        metric_order.append(metric)

    # get datasets
    tfr_train_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    train_ds = return_batched_iter("train", MCd, tfr_train_path)

    tfr_val_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    val_ds = return_batched_iter("train", MCd, tfr_val_path)

    # train loop
    best_val_loss = np.inf
    steps, train_losses, val_losses = [], [], []
    template_str: str = "epoch: {:3} train loss: {:.4f} | val loss: {:.4f}"
    for e in range(MCd["epochs"]):
        # TODO: abstract to fn to clear *all* metrics and loss objects
        avg_train_loss.reset_states()
        avg_val_loss.reset_states()
        for train_metric in train_metric_fns:
            train_metric.reset_states()
        for val_metric in val_metric_fns:
            val_metric.reset_states()

        # logger.debug("-> START iterating training dataset")
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            train_step(
                model,
                x_batch_train,
                y_batch_train,
                loss_object,
                optimizer,
                avg_train_loss,
                train_metric_fns,
            )
        # logger.debug("-> END iterating training dataset")

        # iterate validation after iterating entire training.. this will/should change
        # logger.debug("-> START iterating validation dataset")
        for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
            val_step(
                model,
                x_batch_val,
                y_batch_val,
                loss_object,
                avg_val_loss,
                val_metric_fns,
            )

        # check save best metrics
        cur_val_loss_ = avg_val_loss.result().numpy()
        if cur_val_loss_ < best_val_loss:
            best_val_loss = cur_val_loss_
            model.save_weights(os.path.join(MCd["save_weights_path"]))
            # logger.debug("best params saved: val loss: {:.4f}".format(cur_val_loss))

        # logger.debug("-> END iterating validation dataset")

        # TODO: loop metrics
        cur_train_loss_ = avg_train_loss.result().numpy()
        train_losses.append(cur_train_loss_)
        val_losses.append(cur_val_loss_)
        steps.append(e)
        logger.debug(template_str.format(e + 1, cur_train_loss_, cur_val_loss_))

    logger.info("start creating train_dict")
    return_dict = {}
    # loss history
    return_dict["train_losses"] = train_losses
    return_dict["val_losses"] = val_losses
    return_dict["epochs"] = steps
    # metrics
    for i, name in enumerate(metric_order):
        cur_train_metric_fn = train_metric_fns[i]
        cur_val_metric_fn = val_metric_fns[i]
        return_dict[name] = cur_train_metric_fn.result().numpy()
        return_dict["val_" + name] = cur_val_metric_fn.result().numpy()
    logger.info("[END] creating train_dict")

    return return_dict
