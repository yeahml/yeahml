import os
from typing import Any

import numpy as np
import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metrics import configure_metric
from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging


@tf.function
def eval_step(model, x_batch, y_batch, loss_fn, loss_avg, metric_fns):
    prediction = model(x_batch)
    loss = loss_fn(y_batch, prediction)

    # NOTE: only allow one loss
    loss_avg(loss)

    # TODO: log outputs

    # TODO: ensure pred, gt order
    for eval_metric_fn in metric_fns:
        eval_metric_fn(y_batch, prediction)


def eval_model(
    model: Any,
    meta_cdict,
    log_cdict,
    data_cdict,
    hp_cdict,
    perf_cdict,
    weights_path: str = None,
) -> dict:

    logger = config_logger(meta_cdict["log_dir"], log_cdict, "eval")

    # load best weights
    # TODO: load specific weights according to a param
    if weights_path:
        specified_path = weights_path
    else:
        specified_path = meta_cdict["save_weights_path"]
    model.load_weights(specified_path)
    logger.info(f"params loaded from {specified_path}")

    # loss
    # get loss function
    loss_object = configure_loss(perf_cdict["loss_fn"])

    # mean loss
    avg_eval_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)

    # metrics
    eval_metric_fns, metric_order = [], []
    met_opts = perf_cdict["met_opts_list"]
    for i, metric in enumerate(perf_cdict["met_list"]):
        try:
            met_opt_dict = met_opts[i]
        except TypeError:
            # no options
            met_opt_dict = None
        except IndexError:
            # No options for particular metric
            met_opt_dict = None
            pass
        eval_metric_fn = configure_metric(metric, met_opt_dict)
        eval_metric_fns.append(eval_metric_fn)
        metric_order.append(metric)

    # reset metrics (should already be reset)
    for eval_metric_fn in eval_metric_fns:
        eval_metric_fn.reset_states()

    # get datasets
    tfr_eval_path = os.path.join(data_cdict["TFR_dir"], data_cdict["TFR_train"])
    # TODO: the hp_cdict isn't needed here
    eval_ds = return_batched_iter("eval", data_cdict, hp_cdict, tfr_eval_path)

    logger.info("-> START evaluating model")
    for step, (x_batch, y_batch) in enumerate(eval_ds):
        eval_step(model, x_batch, y_batch, loss_object, avg_eval_loss, eval_metric_fns)
    logger.info("[END] evaluating model")

    logger.info("-> START creating eval_dict")
    eval_dict = {}
    for i, name in enumerate(metric_order):
        cur_metric_fn = eval_metric_fns[i]
        eval_dict[name] = cur_metric_fn.result().numpy()
    logger.info("[END] creating eval_dict")

    # TODO: log each instance

    return eval_dict
