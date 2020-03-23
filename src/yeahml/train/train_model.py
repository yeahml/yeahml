import os
import pathlib
import time
from typing import Any, Dict

import tensorflow as tf

# from yeahml.build.load_params_onto_layer import init_params_from_file  # load params
from yeahml.build.components.loss import configure_loss

# from yeahml.build.components.optimizer import get_optimizer
from yeahml.build.components.metric import configure_metric
from yeahml.build.components.optimizer import return_optimizer
from yeahml.config.model.util import make_hash
from yeahml.dataset.util import get_configured_dataset
from yeahml.log.yf_logging import config_logger  # custom logging


"""
this training logic will need to be cleaned up/refactored a few times, but for
now it is "working" as I hoped.

- create a single fn that accepts "train"/"val" and does the logic (the loops
  are ~the same) and accepts a ds that is of the appropriate size .take()
- simplify/standardize the `return_dict` -- also accept any batch_steps for
  logging + determining when to save params.
- the variable naming throughout this is super inconsistent and not always
  accurate, I intend to improve this over time
- I need to validate the metric/loss/joint logic. I've only tried the present
  case and I'm not sure it will generalize well yet
- tensorboard -- I'm waiting until the loop logic is cleaned up before writing
  to tb again
- early stopping
- save best params
- create a varible in the graph to keep track of the number of batch
  steps/epochs run (if run from a notebook) --- also allow for the passing of an
  epoch param so that if we want to only run ~n more epochs from the notebook we
  can
- I'll also need to ensure the tracking dict is persisted.
"""


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

        return prediction, final_loss, full_losses

    return apply_grad


@tf.function
def train_step(
    model,
    x_batch_train,
    y_batch_train,
    cur_optimizer,
    l2o_objects,
    l2o_loss_record_train,
    joint_loss_record_train,
    metric_objs_train,
    model_apply_grads_fn,
):

    # for i,loss_fn in enumerate(loss_fns):
    prediction, final_loss, full_losses = model_apply_grads_fn(
        model, x_batch_train, y_batch_train, l2o_objects, cur_optimizer
    )

    # from HOML2: NOTE: I'm not sure this belongs here anymore...
    for variable in model.variables:
        if variable.constraint is not None:
            variable.assign(variable.constraint(variable))

    # TODO: only track the mean of the grouped loss
    for loss_rec_name, loss_rec_objects in l2o_loss_record_train.items():
        for i, o in enumerate(loss_rec_objects):
            o.update_state(full_losses[i])

    # NOTE: this could support min/max/mean etc.
    for joint_rec_name, join_rec_obj in joint_loss_record_train.items():
        join_rec_obj.update_state(final_loss)

    # TODO: ensure pred, gt order
    for train_metric in metric_objs_train:
        train_metric.update_state(y_batch_train, prediction)


@tf.function
def val_step(
    model,
    x_batch,
    y_batch,
    l2o_objects,
    l2o_loss_record_val,
    joint_loss_record_val,
    metric_objs_val,
):

    prediction = model(x_batch, training=False)

    # TODO: apply mask?
    full_losses = []
    for loss_fn in l2o_objects:
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

    for loss_rec_name, loss_rec_objects in l2o_loss_record_val.items():
        for i, o in enumerate(loss_rec_objects):
            o.update_state(full_losses[i])

    # NOTE: this could support min/max/mean etc.
    for joint_rec_name, join_rec_obj in joint_loss_record_val.items():
        join_rec_obj.update_state(final_loss)

    # TODO: ensure pred, gt order
    for val_metric in metric_objs_val:
        val_metric.update_state(y_batch, prediction)


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
            loss_type = loss_config["type"]
            avg_train_loss = tf.keras.metrics.Mean(
                name=f"loss_mean_{obj_name}_{loss_type}_train", dtype=tf.float32
            )
            avg_val_loss = tf.keras.metrics.Mean(
                name=f"loss_mean_{obj_name}_{loss_type}_validation", dtype=tf.float32
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
    optimizer_to_loss_name_map = {}
    for cur_optimizer_name, optimizer_dict in optimizers_dict.items():
        loss_names_to_optimize = []
        loss_objs_to_optimize = []
        train_means = []
        val_means = []

        try:
            objectives_to_opt = optimizer_dict["objectives"]
        except KeyError:
            raise KeyError(f"no objectives found for {cur_optimizer_name}")

        in_to_optimizer = None
        for o in objectives_to_opt:
            # add to set of all objectives used - for tracking purposes
            objectives_used.add(o)

            # sanity check ensure loss object from targeted objective exists
            try:
                loss_object = objectives_dict[o]["loss"]["object"]
            except KeyError:
                raise KeyError(f"no loss object is present in objective {o}")

            try:
                train_mean = objectives_dict[o]["loss"]["train_mean"]
            except KeyError:
                raise KeyError(f"no train_mean is present in objective {o}")

            try:
                val_mean = objectives_dict[o]["loss"]["val_mean"]
            except KeyError:
                raise KeyError(f"no val_mean is present in objective {o}")

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
            loss_names_to_optimize.append(o)
            loss_objs_to_optimize.append(loss_object)
            train_means.append(train_mean)
            val_means.append(val_mean)
        # create and include joint metric
        joint_name = "__".join(loss_names_to_optimize) + "__joint"
        train_name = joint_name + "_train"
        val_name = joint_name + "_val"
        joint_object_train = tf.keras.metrics.Mean(name=train_name, dtype=tf.float32)
        joint_object_val = tf.keras.metrics.Mean(name=val_name, dtype=tf.float32)

        optimizer_to_loss_name_map[cur_optimizer_name] = {
            "losses_to_optimize": {
                "names": loss_names_to_optimize,
                "objects": loss_objs_to_optimize,
                "record": {"train": {"mean": train_means}, "val": {"mean": val_means}},
                "joint_name": train_name,
                "joint_record": {
                    "train": {"mean": joint_object_train},
                    "val": {"mean": joint_object_val},
                },
            },
            "in_conf": in_conf,
        }

    # ensure all losses are mapped to an optimizer
    obj_not_used = []
    for obj_name, obj_dict in objectives_dict.items():
        # only add objective if it contains a loss
        try:
            _ = obj_dict["loss"]
            if obj_name not in objectives_used:
                obj_not_used.append(obj_name)
        except KeyError:
            pass
    if obj_not_used:
        raise ValueError(f"objectives {obj_not_used} are not mapped to an optimizer")

    return optimizer_to_loss_name_map


def _map_in_config_to_objective(objectives_dict):
    in_hash_to_objectives = {}
    for o, d in objectives_dict.items():
        in_conf = d["in_config"]
        in_conf_hash = make_hash(in_conf)
        try:
            stored_conf = in_hash_to_objectives[in_conf_hash]["in_config"]
            if not stored_conf == in_conf:
                raise ValueError(
                    f"the hash is the same, but the in config is different..."
                )
        except KeyError:
            in_hash_to_objectives[in_conf_hash] = {"in_config": in_conf}

        # ? is there a case where there is no objective?
        try:
            stored_objectives = in_hash_to_objectives[in_conf_hash]["objectives"]
            stored_objectives.append(o)
        except KeyError:
            in_hash_to_objectives[in_conf_hash]["objectives"] = [o]

    return in_hash_to_objectives


def _create_grouped_metrics(objectives_dict, in_hash_to_objectives):
    in_hash_to_metrics_config = {}

    # loop the different in/out combinations and build metrics for each
    # this dict may become a bit messy because there is the train+val to keep
    # track of
    for k, v in in_hash_to_objectives.items():
        in_hash_to_metrics_config[k] = {"in_config": v["in_config"]}
        in_hash_to_metrics_config[k]["metric_order"] = []
        in_hash_to_metrics_config[k]["objects"] = {"train": [], "val": []}
        for objective in v["objectives"]:
            obj_dict = objectives_dict[objective]
            try:
                cur_metrics = obj_dict["metrics"]
            except KeyError:
                cur_metrics = None

            if cur_metrics:
                in_hash_to_metrics_config[k]["metric_order"].extend(
                    obj_dict["metrics"]["metric_order"]
                )
                in_hash_to_metrics_config[k]["objects"]["train"].extend(
                    obj_dict["metrics"]["train_metrics"]
                )
                in_hash_to_metrics_config[k]["objects"]["val"].extend(
                    obj_dict["metrics"]["val_metrics"]
                )

    return in_hash_to_metrics_config


def _reset_metric_collection(metric_objects):
    # NOTE: I'm not 100% this is always a list
    if isinstance(metric_objects, list):
        for metric_object in metric_objects:
            metric_object.reset_states()
    else:
        metric_objects.reset_states()


# def _full_pass_on_ds():


def _reset_loss_records(loss_dict):
    for name, mets in loss_dict.items():
        if isinstance(mets, list):
            for metric_object in mets:
                metric_object.reset_states()
        else:
            mets.reset_states()


def _record_metrics(
    ds_name, step_name, step_value, perf_dict_tracker, metric_names, mets
):
    # {
    #     "train": {
    #         "meansquarederror": {
    #             "epoch": {"best": None, "steps": None, "values": None}
    #         },
    #         "meanabsoluteerror": {
    #             "epoch": {"best": None, "steps": None, "values": None}
    #         },
    #     },
    #     "val": {
    #         "meansquarederror": {
    #             "epoch": {"best": None, "steps": None, "values": None}
    #         },
    #         "meanabsoluteerror": {
    #             "epoch": {"best": None, "steps": None, "values": None}
    #         },
    #     },
    # }

    best_update = {}
    if not isinstance(mets, list):
        mets = [mets]
    for i, metric_object in enumerate(mets):
        if not perf_dict_tracker[ds_name][metric_names[i]][step_name]["steps"]:
            perf_dict_tracker[ds_name][metric_names[i]][step_name]["steps"] = [
                step_value
            ]
        else:
            perf_dict_tracker[ds_name][metric_names[i]][step_name]["steps"].append(
                step_value
            )

        cur_value = metric_object.result().numpy()
        if not perf_dict_tracker[ds_name][metric_names[i]][step_name]["values"]:
            perf_dict_tracker[ds_name][metric_names[i]][step_name]["values"] = [
                cur_value
            ]
        else:
            perf_dict_tracker[ds_name][metric_names[i]][step_name]["values"].append(
                cur_value
            )

        prev_best = perf_dict_tracker[ds_name][metric_names[i]][step_name]["best"]
        if not prev_best:
            perf_dict_tracker[ds_name][metric_names[i]][step_name]["best"] = cur_value
            update = True
        else:
            if cur_value < prev_best:
                perf_dict_tracker[ds_name][metric_names[i]][step_name][
                    "best"
                ] = cur_value
                update = True
            else:
                update = False
        # uggghhhh.. hardcoded "result"
        best_update[metric_names[i]] = {"result": update}

    return best_update


def _record_losses(
    ds_name, step_name, step_value, loss_dict_tracker, l2o_names, l2o_loss_record
):
    # TODO: I think we should change this logic such that we only keep track of
    # the tensor/metric name and use that as a lookup followed by a dict of
    # additional information?
    best_update = {}

    for name, mets in l2o_loss_record.items():
        # name is the name of {description?} of the metric (mean, etc.)
        if not isinstance(mets, list):
            mets = [mets]
        for i, metric_object in enumerate(mets):

            if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"]:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"] = [
                    step_value
                ]
            else:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "steps"
                ].append(step_value)

            cur_val = metric_object.result().numpy()
            if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"]:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"] = [
                    cur_val
                ]
            else:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "values"
                ].append(cur_val)

            prev_best = loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                "best"
            ]
            if not prev_best:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "best"
                ] = cur_val
                update = True
            else:
                # NOTE: currently assuming min
                if cur_val < prev_best:
                    loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                        "best"
                    ] = cur_val
                    update = True
                else:
                    update = False
            best_update[l2o_names[i]] = {name: update}
            # TODO: logic with the current best

    return best_update


def _record_joint_losses(
    ds_name,
    step_name,
    step_value,
    joint_dict_tracker,
    joint_loss_name,
    joint_loss_record,
):

    # {
    #     "main_obj__second_obj__joint_train": {
    #         "train": {"mean": {"epoch": {"best": None, "steps": None, "values": None}}},
    #         "val": {"mean": {"epoch": {"best": None, "steps": None, "values": None}}},
    #     }
    # }

    best_update = {}
    for name, mets in joint_loss_record.items():
        if not isinstance(mets, list):
            mets = [mets]
        for i, metric_object in enumerate(mets):

            if not joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "steps"
            ]:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "steps"
                ] = [step_value]
            else:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "steps"
                ].append(step_value)

            cur_val = metric_object.result().numpy()
            if not joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "values"
            ]:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "values"
                ] = [cur_val]
            else:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "values"
                ].append(cur_val)

            prev_best = joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                "best"
            ]
            if not prev_best:
                joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                    "best"
                ] = cur_val
                update = True
                best_update[joint_loss_name] = {name: True}
            else:
                # NOTE: currently assuming min
                if cur_val < prev_best:
                    joint_dict_tracker[joint_loss_name][ds_name][name][step_name][
                        "best"
                    ] = cur_val
                    update = True
                else:
                    update = False
            best_update[joint_loss_name] = {name: update}
    return best_update


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

    # create mapping of optimizers to their losses (name, and objects)
    optimizer_to_loss_name_map = _obtain_optimizer_loss_mapping(
        optimizers_dict, objectives_dict
    )

    # create mapping of in_config (same inputs/outputs) to objectives
    in_hash_to_objectives = _map_in_config_to_objective(objectives_dict)

    # use the mapping of in_config to loop and group all metrics -- create
    # groups of metrics to compute at the same time
    # in_hash_to_metrics_config is a mapping of inconfig_hash to inconfig, and
    # metrics (name,train,val)
    in_hash_to_metrics_config = _create_grouped_metrics(
        objectives_dict, in_hash_to_objectives
    )

    # TODO: check that all metrics are accounted for.  If so. raise a not
    # implemented error -- presently the training loop is driven by the
    # optimizers (and as a result all objectives that have matching in_configs).
    # meaning, if a metric does not have a matching in_config, it will not be
    # evaluated.

    # TODO: build best loss dict
    # TODO: this is hardcoded... these "trackers" need to be rethought
    loss_dict_tracker = {}
    for _, temp_dict in optimizer_to_loss_name_map.items():
        for name in temp_dict["losses_to_optimize"]["names"]:
            loss_dict_tracker[name] = {
                "train": {
                    "mean": {"epoch": {"best": None, "steps": None, "values": None}}
                    # "batch": {"best": None, "steps": None, "values": None}
                },
                "val": {
                    "mean": {"epoch": {"best": None, "steps": None, "values": None}}
                    # "batch": {"best": None, "steps": None, "values": None}
                },
            }
            # TODO: if there is another increment to log, do so here

    joint_dict_tracker = {}
    for _, temp_dict in optimizer_to_loss_name_map.items():
        try:
            jd = temp_dict["losses_to_optimize"]["joint_record"]
            joint_name = temp_dict["losses_to_optimize"]["joint_name"]
        except KeyError:
            pass

        if jd:
            joint_dict_tracker[joint_name] = {}
            for ds_name, do in jd.items():
                for description, met_tensor in do.items():
                    joint_dict_tracker[joint_name][ds_name] = {
                        f"{description}": {
                            "epoch": {"best": None, "steps": None, "values": None}
                        }
                    }

    # NOTE: this is out of order from losses:
    # losses = name : ds(train/val): description : .....
    # metrics = ds(train/val) : name : ....
    # I'm not sure which is better yet, but this should be standardized
    perf_dict_tracker = {}
    for _, temp_dict in in_hash_to_metrics_config.items():
        try:
            metric_names = temp_dict["metric_order"]
            md = temp_dict["objects"]
        except KeyError:
            pass

        if md:
            for ds_name, met_list in md.items():
                perf_dict_tracker[ds_name] = {}
                for i, met in enumerate(met_list):
                    perf_dict_tracker[ds_name][metric_names[i]] = {
                        "epoch": {"best": None, "steps": None, "values": None}
                    }

    # TODO: ASSUMPTION: using optimizers sequentially. this may be:
    # - jointly, ordered: sequentially, or unordered: alternate/random

    # TODO: I am not 100% about this logic for maping the optimizer to the
    #   apply_gradient fn... this needs to be confirmed to work as expected
    opt_name_to_gradient_fn = {}
    for cur_optimizer_name, _ in optimizers_dict.items():
        opt_name_to_gradient_fn[cur_optimizer_name] = get_apply_grad_fn()

    # NOTE: I'm not sure looping on epochs makes sense as an outter layer
    # anymore.
    # TODO: is there a way to save a variable in the graph to keep track of
    # epochs (if multiple runs from a notebook?)
    logger.debug("START - iterating epochs dataset")
    all_train_step = 0
    LOGSTEPSIZE = 10
    for e in range(hp_cdict["epochs"]):  #
        logger.debug(f"epoch: {e}")
        for cur_optimizer_name, cur_optimizer_config in optimizers_dict.items():
            # cur_optimizer_config > {"optimizer": tf_obj, "objectives": []}
            # NOTE: if there are multiple objectives, they will be trained *jointly*
            HIST_LOGGED = False  # will update for each optimizer
            logger.debug(f"START - optimizing {cur_optimizer_name}")

            # get optimizer
            cur_optimizer = cur_optimizer_config["optimizer"]

            # get losses
            opt_instructs = optimizer_to_loss_name_map[cur_optimizer_name]
            # opt_instructs = {'ls_to_opt': {'names':[], 'objects': [], in_conf:{}}}
            inhash = make_hash(opt_instructs["in_conf"])
            losses_to_optimize_d = opt_instructs["losses_to_optimize"]
            l2o_names = losses_to_optimize_d["names"]
            l2o_objects = losses_to_optimize_d["objects"]
            l2o_loss_record_train = losses_to_optimize_d["record"]["train"]
            l2o_loss_record_val = losses_to_optimize_d["record"]["val"]
            joint_loss_record_train = losses_to_optimize_d["joint_record"]["train"]
            joint_loss_record_val = losses_to_optimize_d["joint_record"]["val"]
            joint_loss_name = losses_to_optimize_d["joint_name"]

            # get metrics
            metric_collection = in_hash_to_metrics_config[inhash]
            metric_names = metric_collection["metric_order"]
            metric_objs_train = metric_collection["objects"]["train"]
            metric_objs_val = metric_collection["objects"]["val"]

            _reset_metric_collection(metric_objs_train)
            _reset_metric_collection(metric_objs_val)

            # reset states of loss records
            _reset_loss_records(l2o_loss_record_train)
            _reset_loss_records(l2o_loss_record_val)
            _reset_loss_records(joint_loss_record_train)
            _reset_loss_records(joint_loss_record_val)

            cur_apply_grad_fn = opt_name_to_gradient_fn[cur_optimizer_name]

            # TODO: ASSUMPTION: running a full loop over the dataset
            # run full loop on dataset
            logger.debug(f"START iterating training dataset - epoch: {e}")
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                all_train_step += 1

                # TODO: random -- check for nans in loss values

                # track values
                train_step(
                    model,
                    x_batch_train,
                    y_batch_train,
                    cur_optimizer,
                    l2o_objects,
                    l2o_loss_record_train,
                    joint_loss_record_train,
                    metric_objs_train,
                    cur_apply_grad_fn,
                )

                if all_train_step % LOGSTEPSIZE == 0:
                    log_model_params(tr_writer, all_train_step, model)
                    HIST_LOGGED = True

            logger.debug(f"END iterating training dataset- epoch: {e}")

            # TODO: add to tensorboard

            train_best_update = _record_losses(
                "train", "epoch", e, loss_dict_tracker, l2o_names, l2o_loss_record_train
            )
            train_best_joint_update = _record_joint_losses(
                "train",
                "epoch",
                e,
                joint_dict_tracker,
                joint_loss_name,
                joint_loss_record_train,
            )

            train_best_met_update = _record_metrics(
                "train", "epoch", e, perf_dict_tracker, metric_names, metric_objs_train
            )

            # TODO: tensorboard
            # with tr_writer.as_default():
            #     tf.summary.scalar("loss", cur_train_loss_, step=e)
            #     for i, name in enumerate(metric_order):
            #         cur_train_metric_fn = train_metric_fns[i]
            #         tf.summary.scalar(name, cur_train_metric_fn.result().numpy(), step=e)

            # This may not be the place to log these...
            if not HIST_LOGGED:
                log_model_params(tr_writer, all_train_step, model)
                HIST_LOGGED = True

            # iterate validation after iterating entire training.. this will/should
            # change to update on a set frequency -- also, maybe we don't want to
            # run the "full" validation, only a (random) subset?
            logger.debug(f"START iterating validation dataset - epoch: {e}")

            # iterate validation after iterating entire training.. this will/should
            # change to update on a set frequency -- also, maybe we don't want to
            # run the "full" validation, only a (random) subset?
            for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
                val_step(
                    model,
                    x_batch_val,
                    y_batch_val,
                    l2o_objects,
                    l2o_loss_record_val,
                    joint_loss_record_val,
                    metric_objs_val,
                )

            logger.debug(f"END iterating validation dataset - epoch: {e}")

            val_best_update = _record_losses(
                "val", "epoch", e, loss_dict_tracker, l2o_names, l2o_loss_record_val
            )
            val_best_joint_update = _record_joint_losses(
                "val",
                "epoch",
                e,
                joint_dict_tracker,
                joint_loss_name,
                joint_loss_record_val,
            )

            val_best_met_update = _record_metrics(
                "val", "epoch", e, perf_dict_tracker, metric_names, metric_objs_val
            )

            # TODO: save best params with update dict and save params
            # accordingly
            # TODO: use early_stopping:epochs and early_stopping:warmup
            # if cur_val_loss_ < best_val_loss:
            #     if e == 0:
            #         # on the first time params are saved, try to save the model
            #         model.save(save_model_path)
            #         logger.debug(f"model saved to: {save_model_path}")
            #     best_val_loss = cur_val_loss_
            #     model.save_weights(save_best_param_path)

            #     logger.debug(f"best params saved: val loss:
            #     {cur_val_loss_:.4f}")

            # TODO: log epoch results
            # logger.debug()

            # TODO: tensorboard
            # with v_writer.as_default():
            #     tf.summary.scalar("loss", cur_val_loss_, step=e)
            #     for i, name in enumerate(metric_order):
            #         cur_val_metric_fn = val_metric_fns[i]
            #         tf.summary.scalar(name, cur_val_metric_fn.result().numpy(), step=e)

    # TODO: I think the 'joint' should likely be the optimizer name, not the
    # combination of losses name, this would also simplify the creation of these
    return_dict = {
        "loss": loss_dict_tracker,
        "joint": joint_dict_tracker,
        "metrics": perf_dict_tracker,
    }

    return return_dict
