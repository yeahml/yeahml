import os
import pathlib
import time
from typing import Any, Dict

import numpy as np
import tensorflow as tf

# from yeahml.build.load_params_onto_layer import init_params_from_file  # load params
from yeahml.build.components.loss import configure_loss

# from yeahml.build.components.optimizer import get_optimizer
from yeahml.build.components.metrics import configure_metric
from yeahml.build.components.optimizer import return_optimizer
from yeahml.dataset.util import get_datasets_from_tfrs
from yeahml.log.yf_logging import config_logger  # custom logging


def get_apply_grad_fn():
    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    @tf.function
    def apply_grad(model, x_batch, y_batch, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            prediction = model(x_batch, training=True)

            # TODO: apply mask?
            loss = loss_fn(y_batch, prediction)

            # TODO: custom weighting for training could be applied here
            # weighted_losses = loss * weights_per_instance
            main_loss = tf.reduce_mean(loss)

            # model.losses contains the kernel/bias constrains/regularizers
            full_loss = tf.add_n([main_loss] + model.losses)

        grads = tape.gradient(full_loss, model.trainable_variables)

        # NOTE: any gradient adjustments would happen here
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return prediction, full_loss

    return apply_grad


@tf.function
def train_step(
    model, x_batch, y_batch, loss_fn, optimizer, loss_avg, metrics, model_apply_grads_fn
):

    prediction, loss = model_apply_grads_fn(model, x_batch, y_batch, loss_fn, optimizer)

    # from HOML2:
    for variable in model.variables:
        if variable.constraint is not None:
            variable.assign(variable.constraint(variable))

    # NOTE: only allow one loss? I don't see why only one loss should be allowed?
    loss_avg(loss)

    # TODO: ensure pred, gt order
    for train_metric in metrics:
        train_metric(y_batch, prediction)


@tf.function
def val_step(model, x_batch, y_batch, loss_fn, loss_avg, metrics):

    prediction = model(x_batch, training=False)
    loss = loss_fn(y_batch, prediction)

    # NOTE: only allow one loss
    loss_avg(loss)

    # TODO: ensure pred, gt order
    for val_metric in metrics:
        val_metric(y_batch, prediction)


def log_model_params(tr_writer, g_train_step, model):
    with tr_writer.as_default():
        for v in model.variables:
            tf.summary.histogram(v.name.split(":")[0], v.numpy(), step=g_train_step)


def get_exp_time():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return run_id


def train_model(
    model: Any, config_dict: Dict[str, Dict[str, Any]], datasets: tuple = ()
) -> Dict[str, Any]:

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]
    hp_cdict: Dict[str, Any] = config_dict["hyper_parameters"]
    perf_cdict: Dict[str, Any] = config_dict["performance"]

    return_dict = {}

    logger = config_logger(model_cdict["model_root_dir"], log_cdict, "train")
    logger.info("-> START training graph")

    # save run specific information
    exp_time = get_exp_time()
    # model
    model_run_path = os.path.join(model_cdict["save/model"], exp_time)
    pathlib.Path(model_run_path).mkdir(parents=True, exist_ok=True)
    save_model_path = os.path.join(model_run_path, "model.h5")
    # params
    param_run_path = os.path.join(model_cdict["save/params"], exp_time)
    pathlib.Path(param_run_path).mkdir(parents=True, exist_ok=True)
    save_best_param_path = os.path.join(param_run_path, "best_params.h5")
    # Tensorboard
    # TODO: eventually, this needs to be flexible enough to allow for new writes
    # every n steps
    tb_logdir = os.path.join(model_cdict["tf_logs"], exp_time)
    pathlib.Path(tb_logdir).mkdir(parents=True, exist_ok=True)
    tr_writer = tf.summary.create_file_writer(os.path.join(tb_logdir, "train"))
    v_writer = tf.summary.create_file_writer(os.path.join(tb_logdir, "val"))

    # get model
    # get optimizer

    # TODO: config optimizer (follow template for losses)
    optim_dict = return_optimizer(hp_cdict["optimizer_dict"]["type"])
    optimizer = optim_dict["function"]

    # configure optimizer
    temp_dict = hp_cdict["optimizer_dict"].copy()
    temp_dict.pop("type")
    optimizer = optimizer(**temp_dict)

    # get loss function
    loss_object = configure_loss(perf_cdict["loss_fn"])

    # mean loss
    avg_train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)

    # get metrics
    train_metric_fns = []
    val_metric_fns = []
    metric_order = []
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
        train_metric_fn = configure_metric(metric, met_opt_dict)
        train_metric_fns.append(train_metric_fn)
        val_metric_fn = configure_metric(metric, met_opt_dict)
        val_metric_fns.append(val_metric_fn)
        metric_order.append(metric)

    # get datasets
    # TODO: there needs to be some check here to ensure the same datsets are being compared.
    if not datasets:
        train_ds, val_ds = get_datasets_from_tfrs(data_cdict, hp_cdict)
    else:
        assert (
            len(datasets) == 2
        ), f"{len(datasets)} datasets were passed, please pass 2 datasets (train, validation)"
        train_ds, val_ds = datasets

    # # write graph
    # g_writer = tf.summary.create_file_writer(os.path.join(tb_logdir, "graph"))
    # prof_ds = train_ds.take(2)
    # # tf.summary.trace_on(graph=True, profiler=True)
    # for (x_batch, _) in prof_ds:
    #     _ = model(x_batch, training=False)
    #     # with g_writer.as_default():
    #     #     tf.summary.trace_export(
    #     #         "dataset_input",
    #     #         step=0,
    #     #         profiler_outdir=os.path.join(tb_logdir, "graph"),
    #     #     )
    #     tf.summary.trace_on(graph=True, profiler=True)
    #     _ = model(x_batch, training=False)
    #     with g_writer.as_default():
    #         tf.summary.trace_export(
    #             "model_inference",
    #             step=0,
    #             profiler_outdir=os.path.join(tb_logdir, "graph"),
    #         )
    # print("done profile")

    # train loop
    apply_grad_fn = get_apply_grad_fn()
    # TODO: remove np dependency
    best_val_loss = np.inf
    steps, train_losses, val_losses = [], [], []
    template_str: str = "epoch: {:3} train loss: {:.4f} | val loss: {:.4f}"
    for e in range(hp_cdict["epochs"]):
        # TODO: abstract to fn to clear *all* metrics and loss objects
        avg_train_loss.reset_states()
        avg_val_loss.reset_states()
        for train_metric in train_metric_fns:
            train_metric.reset_states()
        for val_metric in val_metric_fns:
            val_metric.reset_states()

        logger.debug("-> START iterating training dataset")
        g_train_step = 0
        LOGSTEPSIZE = 10
        HIST_LOGGED = False
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            g_train_step += 1

            train_step(
                model,
                x_batch_train,
                y_batch_train,
                loss_object,
                optimizer,
                avg_train_loss,
                train_metric_fns,
                apply_grad_fn,
            )
            if g_train_step % LOGSTEPSIZE == 0:
                HIST_LOGGED = True
                log_model_params(tr_writer, g_train_step, model)

        if not HIST_LOGGED:
            # in case there are not LOGSTEPSIZE in the training set
            log_model_params(tr_writer, g_train_step, model)

        logger.debug("-> END iterating training dataset")

        # iterate validation after iterating entire training.. this will/should change
        logger.debug("-> START iterating validation dataset")
        for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
            val_step(
                model,
                x_batch_val,
                y_batch_val,
                loss_object,
                avg_val_loss,
                val_metric_fns,
            )

        logger.debug("-> END iterating validation dataset")

        # check save best metrics

        cur_val_loss_ = avg_val_loss.result().numpy()
        if cur_val_loss_ < best_val_loss:
            # TODO: remove np dependency
            if best_val_loss == np.inf:
                # on the first time params are saved, try to save the model
                model.save(save_model_path)
                logger.debug(f"model saved to: {save_model_path}")
            best_val_loss = cur_val_loss_
            model.save_weights(save_best_param_path)

            logger.debug(f"best params saved: val loss: {cur_val_loss_:.4f}")

        # TODO: loop metrics
        cur_train_loss_ = avg_train_loss.result().numpy()
        train_losses.append(cur_train_loss_)
        val_losses.append(cur_val_loss_)
        steps.append(e)
        logger.debug(template_str.format(e + 1, cur_train_loss_, cur_val_loss_))

        with tr_writer.as_default():
            tf.summary.scalar("loss", cur_train_loss_, step=e)
            for i, name in enumerate(metric_order):
                cur_train_metric_fn = train_metric_fns[i]
                tf.summary.scalar(name, cur_train_metric_fn.result().numpy(), step=e)

        with v_writer.as_default():
            tf.summary.scalar("loss", cur_val_loss_, step=e)
            for i, name in enumerate(metric_order):
                cur_val_metric_fn = val_metric_fns[i]
                tf.summary.scalar(name, cur_val_metric_fn.result().numpy(), step=e)

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
