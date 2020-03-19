import math
import os
import pathlib
import time
from typing import Any, Dict
from yeahml.config.model.util import make_hash

import tensorflow as tf

# from yeahml.build.load_params_onto_layer import init_params_from_file  # load params
from yeahml.build.components.loss import configure_loss

# from yeahml.build.components.optimizer import get_optimizer
from yeahml.build.components.metric import configure_metric
from yeahml.build.components.optimizer import return_optimizer
from yeahml.dataset.util import get_configured_dataset
from yeahml.log.yf_logging import config_logger  # custom logging
import sys


def get_apply_grad_fn():

    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    @tf.function
    def apply_grad(model, x_batch, y_batch, loss_fns, optimizer):
        with tf.GradientTape() as tape:
            prediction = model(x_batch, training=True)

            # TODO: apply mask?
            full_losses = []
            for loss_fn in loss_fns:
                loss = loss_fn(y_batch, prediction)

                # TODO: custom weighting for training could be applied here
                # weighted_losses = loss * weights_per_instance
                main_loss = tf.reduce_mean(loss)
                # model.losses contains the kernel/bias constraints/regularizers
                cur_loss = tf.add_n([main_loss] + model.losses)
                # full_loss = tf.add_n(full_loss, cur_loss)
                full_losses.append(cur_loss)
                # create joint loss for current optimizer
                # e.g. final_loss = tf.reduce_mean(loss1 + loss2)
            final_loss = tf.reduce_mean(tf.math.add_n(full_losses))

        # TODO: maybe we should be able to specify which params to be optimized
        # by specific optimizers
        grads = tape.gradient(final_loss, model.trainable_variables)

        # NOTE: any gradient adjustments would happen here
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return prediction, final_loss

    return apply_grad


@tf.function
def train_step(
    model,
    x_batch,
    y_batch,
    loss_fns,
    optimizer,
    loss_avg,
    metrics,
    model_apply_grads_fn,
):

    # for i,loss_fn in enumerate(loss_fns):
    prediction, loss = model_apply_grads_fn(
        model, x_batch, y_batch, loss_fns, optimizer
    )

    # from HOML2: NOTE: I'm not sure this belongs here anymore...
    for variable in model.variables:
        if variable.constraint is not None:
            variable.assign(variable.constraint(variable))

    # TODO: only track the mean of the grouped loss
    loss_avg[0](loss)
    # print(loss_avg[0].result())

    # TODO: ensure pred, gt order
    for train_metric in metrics:
        train_metric(y_batch, prediction)


@tf.function
def val_step(model, x_batch, y_batch, loss_fns, loss_avgs, metrics):

    prediction = model(x_batch, training=False)

    all_losses = []
    for i, loss_fn in enumerate(loss_fns):
        loss = loss_fn(y_batch, prediction)
        all_losses.append(loss)

        loss_avgs[i](loss)
    # TODO: this also needs to be logged
    # final_loss = tf.reduce_mean(tf.math.add_n(all_losses))

    # TODO: ensure pred, gt order
    for val_metric in metrics:
        val_metric(y_batch, prediction)


def log_model_params(tr_writer, g_train_step, model):
    with tr_writer.as_default():
        for v in model.variables:
            tf.summary.histogram(v.name.split(":")[0], v.numpy(), step=g_train_step)


def _model_run_path(full_exp_path):
    # save run specific information
    exp_time = time.strftime("run_%Y_%m_%d-%H_%M_%S")

    # experiment/model
    model_path = full_exp_path.joinpath("model")

    # model/experiment_time
    model_run_path = model_path.joinpath(exp_time)
    model_run_path.mkdir(parents=True, exist_ok=True)

    return model_run_path


def _create_model_training_paths(model_run_path):
    # model/exp_time/save/
    run_save = model_run_path.joinpath("save")
    run_save.mkdir(parents=True, exist_ok=True)

    # model/exp_time/save/params
    param_run_path = run_save.joinpath("params")
    param_run_path.mkdir(parents=True, exist_ok=True)

    # model/exp_time/save/model.h5
    save_model_path = str(run_save.joinpath("model.h5"))
    # model/exp_time/save/params/<specific_params>.h5
    save_best_param_path = str(param_run_path.joinpath("best_params.h5"))

    return save_model_path, save_best_param_path


def _get_tb_writers(model_run_path):
    # Tensorboard
    # TODO: eventually, this needs to be flexible enough to allow for new writes
    # every n steps
    tb_logdir = model_run_path.joinpath("tf_logs")
    tb_logdir.mkdir(parents=True, exist_ok=True)
    tr_writer = tf.summary.create_file_writer(os.path.join(tb_logdir, "train"))
    v_writer = tf.summary.create_file_writer(os.path.join(tb_logdir, "val"))

    return tr_writer, v_writer


def _get_metrics(metric_config):
    train_metric_fns = []
    val_metric_fns = []
    metric_order = []

    assert len(metric_config["options"]) == len(
        metric_config["type"]
    ), f"len of options does not len of metrics: {len(metric_config['options'])} != {len(metric_config['type'])}"

    # loop operations and options
    try:
        met_opts = metric_config["options"]
    except KeyError:
        met_opts = None
    for i, metric in enumerate(metric_config["type"]):
        if met_opts:
            met_opt_dict = met_opts[i]
        else:
            met_opt_dict = None

        # train
        train_metric_fn = configure_metric(metric, met_opt_dict)
        train_metric_fns.append(train_metric_fn)

        # validation
        val_metric_fn = configure_metric(metric, met_opt_dict)
        val_metric_fns.append(val_metric_fn)

        # order
        metric_order.append(metric)

    return (metric_order, train_metric_fns, val_metric_fns)


def _get_objectives(objectives):
    # TODO: should these be grouped based on their inputs?

    obj_conf = {}
    for obj_name, config in objectives.items():
        in_config = config["in_config"]

        try:
            loss_config = config["loss"]
        except KeyError:
            loss_config = None

        try:
            metric_config = config["metric"]
        except KeyError:
            metric_config = None

        if not loss_config and not metric_config:
            raise ValueError(f"Neither a loss or metric was defined for {obj_name}")

        if loss_config:
            loss_object = configure_loss(loss_config)

            # mean loss for both training and validation
            # NOTE: maybe it doesn't make sense to add this here... this could
            # instead be created when grouping the metrics.
            avg_train_loss = tf.keras.metrics.Mean(name="train_loss", dtype=tf.float32)
            avg_val_loss = tf.keras.metrics.Mean(
                name="validation_loss", dtype=tf.float32
            )
        else:
            loss_object, avg_train_loss, avg_val_loss = None, None, None

        if metric_config:
            metric_order, train_metric_fns, val_metric_fns = _get_metrics(metric_config)
        else:
            metric_order, train_metric_fns, val_metric_fns = None, None, None

        obj_conf[obj_name] = {
            "in_config": in_config,
            "loss": {
                "object": loss_object,
                "train_mean": avg_train_loss,
                "val_mean": avg_val_loss,
            },
            "metrics": {
                "metric_order": metric_order,
                "train_metrics": train_metric_fns,
                "val_metrics": val_metric_fns,
            },
        }

    # Currently, only supervised is accepted
    for obj_name, obj_dict in obj_conf.items():
        if obj_dict["in_config"]["type"] != "supervised":
            raise NotImplementedError(
                f"only 'supervised' is accepted as the type for the in_config of {obj_name}, not {obj_conf['in_config']['type']} yet..."
            )

    return obj_conf


def _get_optimizers(optim_cdict):
    def _configure_optimizer(opt_dict):
        # TODO: this should not be here. (follow template for losses)
        optim_dict = return_optimizer(opt_dict["type"])
        optimizer = optim_dict["function"]

        # configure optimizers
        temp_dict = opt_dict.copy()
        optimizer = optimizer(**temp_dict["options"])

        return optimizer

    optimizers_dict = {}
    for opt_name, opt_dict in optim_cdict["optimizers"].items():
        configured_optimizer = _configure_optimizer(opt_dict)
        optimizers_dict[opt_name] = {
            "optimizer": configured_optimizer,
            "objectives": opt_dict["objectives"],
        }

    return optimizers_dict


def _get_datasets(datasets, data_cdict, hp_cdict):
    # TODO: there needs to be some check here to ensure the same datsets are being compared.
    if not datasets:
        train_ds = get_configured_dataset(
            data_cdict, hp_cdict, ds=None, ds_type="train"
        )
        val_ds = get_configured_dataset(data_cdict, hp_cdict, ds=None, ds_type="val")
    else:
        # TODO: apply shuffle/aug/reshape from config
        assert (
            len(datasets) == 2
        ), f"{len(datasets)} datasets were passed, please pass 2 datasets (train, validation)"
        train_ds, val_ds = datasets
        train_ds = get_configured_dataset(data_cdict, hp_cdict, ds=train_ds)
        val_ds = get_configured_dataset(data_cdict, hp_cdict, ds=val_ds)

    return train_ds, val_ds


def _obtain_optimizer_loss_mapping(optimizers_dict, objectives_dict):
    # NOTE: multiple losses by the same optimizer, are currently only modeled
    # jointly, if we wish to model the losses seperately (sequentially or
    # alternating), then we would want to use a second optimizer
    objectives_used = set()
    optimizer_loss_name_map = {}
    for optimizer_name, optimizer_dict in optimizers_dict.items():
        losses_to_optimize = []

        try:
            objectives_to_opt = optimizer_dict["objectives"]
        except KeyError:
            raise KeyError(f"no objectives found for {optimizer_name}")

        in_to_optimizer = None
        for o in objectives_to_opt:
            # add to set of all objectives used - for tracking purposes
            objectives_used.add(o)

            # sanity check ensure loss object from targeted objective exists
            try:
                _ = objectives_dict[o]["loss"]["object"]
            except KeyError:
                raise KeyError(f"no loss object is present in objective {o}")

            try:
                in_conf = objectives_dict[o]["in_config"]
            except NotImplementedError:
                raise NotImplementedError(
                    f"no options present in {objectives_dict[o]['in_config']}"
                )

            if in_to_optimizer:
                if not in_to_optimizer == in_conf:
                    raise ValueError(
                        f"The in to optimizer is {in_to_optimizer} but the in_conf for {o} is {in_conf}, they should be the same"
                    )
            else:
                in_to_optimizer = in_conf

            # add loss object to a list for grouping
            losses_to_optimize.append(o)

        optimizer_loss_name_map[optimizer_name] = {
            "losses_to_optimize": losses_to_optimize,
            "in_conf": in_conf,
        }

    # ensure all losses are mapped to an optimizer
    obj_not_used = []
    for obj_name, obj_dict in objectives_dict.items():
        # only add objective if it contains a loss
        try:
            l = obj_dict["loss"]
            if obj_name not in objectives_used:
                obj_not_used.append(obj_name)
        except KeyError:
            pass
    if obj_not_used:
        raise ValueError(f"objectives {obj_not_used} are not mapped to an optimizer")

    return optimizer_loss_name_map


def _map_in_config_to_objective(objectives_dict):
    in_hash_to_conf = {}
    for o, d in objectives_dict.items():
        in_conf = d["in_config"]
        in_conf_hash = make_hash(in_conf)
        try:
            stored_conf = in_hash_to_conf[in_conf_hash]["in_config"]
            if not stored_conf == in_conf:
                raise ValueError(
                    f"the hash is the same, but the in config is different..."
                )
        except KeyError:
            in_hash_to_conf[in_conf_hash] = {"in_config": in_conf}

        # ? is there a case where there is no objective?
        try:
            stored_objectives = in_hash_to_conf[in_conf_hash]["objectives"]
            stored_objectives.append(o)
        except KeyError:
            in_hash_to_conf[in_conf_hash]["objectives"] = [o]

    return in_hash_to_conf


def _create_grouped_metrics(objectives_dict, in_hash_to_conf):
    grouped_metrics = {}

    # loop the different in/out combinations and build metrics for each
    # this dict may become a bit messy because there is the train+val to keep
    # track of
    for k, v in in_hash_to_conf.items():
        grouped_metrics[k] = {"in_config": v["in_config"]}
        for objective in v["objectives"]:
            obj_dict = objectives_dict[objective]
            grouped_metrics[k]["jack"] = {}
            print(obj_dict)
            print("-----")

    return grouped_metrics


def train_model(
    model: Any, config_dict: Dict[str, Dict[str, Any]], datasets: tuple = ()
) -> Dict[str, Any]:

    # unpack configurations
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]
    hp_cdict: Dict[str, Any] = config_dict["hyper_parameters"]

    perf_cdict: Dict[str, Any] = config_dict["performance"]
    optim_cdict: Dict[str, Any] = config_dict["optimize"]

    return_dict = {}

    full_exp_path = (
        pathlib.Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
    )
    logger = config_logger(full_exp_path, log_cdict, "train")
    logger.info("-> START training graph")

    # build paths and obtain tb writers
    model_run_path = _model_run_path(full_exp_path)
    save_model_path, save_best_param_path = _create_model_training_paths(model_run_path)
    tr_writer, v_writer = _get_tb_writers(model_run_path)

    # get datasets
    train_ds, val_ds = _get_datasets(datasets, data_cdict, hp_cdict)

    optimizers_dict = _get_optimizers(optim_cdict)
    objectives_dict = _get_objectives(perf_cdict["objectives"])

    # TODO: We need to be able to specify whether the losses should be separately
    # or jointly combined.

    # create mapping of optimizers to their losses
    optimizer_loss_name_map = _obtain_optimizer_loss_mapping(
        optimizers_dict, objectives_dict
    )

    # create mapping of in_config (same inputs/outputs) to objectives
    in_hash_to_conf = _map_in_config_to_objective(objectives_dict)

    # TODO: loop and group all metrics -- create groups of metrics to compute
    # at the same time

    print(optimizer_loss_name_map)
    print("-----")
    print(objectives_dict)
    print("------")
    print(in_hash_to_conf)

    print("========" * 10)

    grouped_metrics = _create_grouped_metrics(objectives_dict, in_hash_to_conf)
    print("========" * 10)
    print(grouped_metrics)

    sys.exit("done")

    # TODO: group objectives by the in/

    # train loop
    apply_grad_fn = get_apply_grad_fn()
    best_val_loss = math.inf
    # TODO: hardcoded
    steps = []  # train_losses, val_losses = [], ([], []), ([], [])
    template_str: str = "epoch: {:3} train loss: {:.4f} | val loss: {:.4f}"
    for e in range(hp_cdict["epochs"]):
        # TODO: abstract to fn to clear *all* metrics and loss objects
        # avg_train_loss.reset_states()
        # avg_val_loss.reset_states()
        # for train_metric in train_metric_fns:
        #     train_metric.reset_states()
        # for val_metric in val_metric_fns:
        #     val_metric.reset_states()

        logger.debug("-> START iterating training dataset")
        g_train_step = 0
        LOGSTEPSIZE = 10
        HIST_LOGGED = False
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            g_train_step += 1
            # TODO: sequential vs alternate

            for (
                cur_optimizer_name,
                cur_optimizer_dict,
            ) in optimizer_loss_name_map.items():

                objectives_to_optimize = cur_optimizer_dict["losses_to_optimize"]

                cur_optimizer = optimizers_dict[cur_optimizer_name]["optimizer"]

                # TODO: these need to be grouped before hand
                metric_names_during_optimization = []
                metrics_during_optimization_train = []
                metrics_during_optimization_val = []
                # TODO: there needs to be a `smarter` way to group all these
                # metrics -- likely based on the inputs/outputs being used
                loss_fns = []
                loss_means_train = []
                loss_means_val = []
                for o_to_o in objectives_to_optimize:
                    loss_fns.append(objectives_dict[o_to_o]["loss"]["object"])
                    loss_means_train.append(
                        objectives_dict[o_to_o]["loss"]["train_mean"]
                    )
                    loss_means_val.append(objectives_dict[o_to_o]["loss"]["val_mean"])
                    if objectives_dict[o_to_o]["metrics"]:
                        # only if the grouping has metrics
                        metric_names_during_optimization.extend(
                            objectives_dict[o_to_o]["metrics"]["metric_order"]
                        )
                        metrics_during_optimization_train.extend(
                            objectives_dict[o_to_o]["metrics"]["train_metrics"]
                        )
                        metrics_during_optimization_val.extend(
                            objectives_dict[o_to_o]["metrics"]["val_metrics"]
                        )

                # for m_during_opt in
                train_step(
                    model,
                    x_batch_train,
                    y_batch_train,
                    loss_fns,
                    cur_optimizer,
                    loss_means_train,
                    metrics_during_optimization_train,
                    apply_grad_fn,
                )

            if g_train_step % LOGSTEPSIZE == 0:
                HIST_LOGGED = True
                log_model_params(tr_writer, g_train_step, model)

        if not HIST_LOGGED:
            # in case there are not LOGSTEPSIZE in the training set
            log_model_params(tr_writer, g_train_step, model)

        logger.debug("-> END iterating training dataset")

        # iterate validation after iterating entire training.. this will/should
        # change to update on a set frequency -- also, maybe we don't want to
        # run the "full" validation, only a (random) subset?
        logger.debug("-> START iterating validation dataset")

        for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
            val_step(
                model,
                x_batch_val,
                y_batch_val,
                loss_fns,
                loss_means_val,
                metrics_during_optimization_val,
            )

        logger.debug("-> END iterating validation dataset")

        # check save best metrics.. this is going to get a little hairy. we need
        # to keep track of the `best_losses` for multiple losses

        pls = []
        for i, cv in enumerate(loss_means_val):
            pls.append(cv.result().numpy())
        print(pls)

        # TODO: use early_stopping:epochs and early_stopping:warmup
        # if cur_val_loss_ < best_val_loss:
        #     if e == 0:
        #         # on the first time params are saved, try to save the model
        #         model.save(save_model_path)
        #         logger.debug(f"model saved to: {save_model_path}")
        #     best_val_loss = cur_val_loss_
        #     model.save_weights(save_best_param_path)

        #     logger.debug(f"best params saved: val loss: {cur_val_loss_:.4f}")

        # TODO: loop metrics
        # cur_train_loss_ = avg_train_loss.result().numpy()
        # train_losses.append(cur_train_loss_)
        # val_losses.append(cur_val_loss_)
        steps.append(e)
        # logger.debug(template_str.format(e + 1, cur_train_loss_, cur_val_loss_))

        # with tr_writer.as_default():
        #     tf.summary.scalar("loss", cur_train_loss_, step=e)
        #     for i, name in enumerate(metric_order):
        #         cur_train_metric_fn = train_metric_fns[i]
        #         tf.summary.scalar(name, cur_train_metric_fn.result().numpy(), step=e)

        # with v_writer.as_default():
        #     tf.summary.scalar("loss", cur_val_loss_, step=e)
        #     for i, name in enumerate(metric_order):
        #         cur_val_metric_fn = val_metric_fns[i]
        #         tf.summary.scalar(name, cur_val_metric_fn.result().numpy(), step=e)

    logger.info("start creating train_dict")
    # return_dict = {}

    # loss history
    # return_dict["train_losses"] = train_losses
    # return_dict["val_losses"] = val_losses
    # return_dict["epochs"] = steps

    # metrics
    # for i, name in enumerate(metric_order):
    #     cur_train_metric_fn = train_metric_fns[i]
    #     cur_val_metric_fn = val_metric_fns[i]
    #     return_dict[name] = cur_train_metric_fn.result().numpy()
    #     return_dict["val_" + name] = cur_val_metric_fn.result().numpy()
    # logger.info("[END] creating train_dict")

    return return_dict
