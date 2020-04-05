import pathlib
from typing import Any, Dict

import tensorflow as tf

from yeahml.config.model.util import make_hash
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.train.setup.datasets import get_datasets
from yeahml.train.setup.loop_dynamics import (  # obtain_optimizer_loss_mapping,
    create_grouped_metrics,
    get_optimizers,
    map_in_config_to_objective,
)
from yeahml.train.setup.objectives import get_objectives
from yeahml.train.setup.paths import (
    create_model_run_path,
    create_model_training_paths,
    get_tb_writers,
)
from yeahml.train.setup.tracker.loss import create_loss_trackers, update_loss_trackers
from yeahml.train.setup.tracker.metric import (
    create_metric_trackers,
    update_metric_trackers,
)
from yeahml.train.setup.tracker.tracker import (
    create_joint_dict_tracker,
    record_joint_losses,
)

# from yeahml.build.load_params_onto_layer import init_params_from_file  # load params


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
- create a variable in the graph to keep track of the number of batch
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


def _reset_metric_collection(metric_objects):
    # NOTE: I'm not 100% this is always a list
    if isinstance(metric_objects, list):
        for metric_object in metric_objects:
            metric_object.reset_states()
    else:
        metric_objects.reset_states()


def _reset_loss_records(loss_dict):
    for name, mets in loss_dict.items():
        if isinstance(mets, list):
            for metric_object in mets:
                metric_object.reset_states()
        else:
            mets.reset_states()


def train_model(
    model: Any, config_dict: Dict[str, Dict[str, Any]], datasets: tuple = ()
) -> Dict[str, Any]:

    # TODO: option to reinitialize model?

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
    model_run_path = create_model_run_path(full_exp_path)
    save_model_path, save_best_param_path = create_model_training_paths(model_run_path)
    tr_writer, v_writer = get_tb_writers(model_run_path)

    # get datasets
    train_ds, val_ds = get_datasets(datasets, data_cdict, hp_cdict)

    # TODO: this section is ...messy
    #########################################################################
    DATASETS = ["train", "val"]

    # {optimizer_name: {"optimizer": tf.obj, "objective": [objective_name]}}
    optimizers_dict = get_optimizers(optim_cdict)

    # {objective_name: "in_config": {...}, "loss": {...}, "metric": {...}}
    objectives_dict = get_objectives(perf_cdict["objectives"], dataset_names=DATASETS)

    # TODO: training_directive may be empty.
    # {
    #     "YEAHML_1": {"optimizers": ["YEAHML_0", "second_opt"], "operation": "&"},
    #     "YEAHML_0": {"optimizers": ["main_opt", "second_opt"], "operation": ","},
    # }
    training_directive = optim_cdict["directive"]

    # create mapping of optimizers to their losses (name, and objects)
    # optimizer_to_loss_name_map = obtain_optimizer_loss_mapping(
    #     optimizers_dict, objectives_dict, datasets=DATASETS
    # )

    # NOTE:
    # now that we have the training directive, we need to group the losses and
    # metrics such that we can access them easily during training
    # 1. group metrics based on in/out similarity (idk if this always holds)
    # 2. match the grouped metrics to when the optimizers are run
    # ---
    # a. create a "joint"/average loss for each instruction step
    # b. create a tracker for this
    # ---
    # ensure these (loss, joint loss, and metrics) can all be easily accessed
    # xxxxxxxxxxxxxxxxxxxxx
    # sort the keys for the instructions
    # train in order
    # TODO: per batch/epoch/adaptive -- all open questions.

    # create mapping of in_config (same inputs/outputs) to objectives

    in_hash_to_objectives = map_in_config_to_objective(objectives_dict)

    # create groups of metrics to compute at the same time

    in_hash_to_metrics_config = create_grouped_metrics(
        objectives_dict, in_hash_to_objectives
    )

    print("#####" * 8)
    print(training_directive)
    print("------" * 8)
    print(in_hash_to_objectives)
    print("#####" * 8)

    # TODO: check that all metrics are accounted for.  If so. raise a not
    # implemented error -- presently the training loop is driven by the
    # optimizers (and as a result all objectives that have matching in_configs).
    # meaning, if a metric does not have a matching in_config, it will not be
    # evaluated.

    #########################################################################

    # TODO: build best loss dict
    # TODO: this is hardcoded... these "trackers" need to be rethought
    # optimizer_to_loss_name_map, ds_names=None, descriptions=None,
    # to_track=None
    # TODO: the `to_track` should be infered from the config.. that is not all
    # losses should be minimized -- will need to ensure optimizer op also
    # considers this.
    loss_trackers = create_loss_trackers(
        optimizer_to_loss_name_map,
        ds_names=DATASETS,
        descriptions=["mean"],
        to_track=["min"],
    )
    joint_dict_tracker = create_joint_dict_tracker(optimizer_to_loss_name_map)

    # TODO: the to_track should be specific to the metric
    metric_trackers = create_metric_trackers(
        in_hash_to_metrics_config, to_track=["max"]
    )

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
    # TODO: this total_train_inst is the number of instances the model has
    # trained on and in stop optimizer specific. there should be multiple
    # values, one for each otimizer and a total_steps
    total_train_inst = 0
    for e in range(hp_cdict["epochs"]):  #
        logger.debug(f"epoch: {e}")
        for cur_optimizer_name, cur_optimizer_config in optimizers_dict.items():
            # cur_optimizer_config > {"optimizer": tf_obj, "objectives": []}
            # NOTE: if there are multiple objectives, they will be trained *jointly*
            HIST_LOGGED = False  # will update for each optimizer
            logger.debug(f"START - optimizing {cur_optimizer_name}")

            # get optimizer
            cur_optimizer = cur_optimizer_config["optimizer"]

            # get losses (optimizer specific)
            opt_instructs = optimizer_to_loss_name_map[cur_optimizer_name]
            # opt_instructs = {'ls_to_opt': {'names':[], 'objects': [], in_conf:{}}}
            losses_to_optimize_d = opt_instructs["losses_to_optimize"]
            l2o_names = losses_to_optimize_d["names"]
            l2o_objects = losses_to_optimize_d["objects"]
            l2o_loss_record_train = losses_to_optimize_d["record"]["train"]
            l2o_loss_record_val = losses_to_optimize_d["record"]["val"]
            joint_loss_record_train = losses_to_optimize_d["joint_record"]["train"]
            joint_loss_record_val = losses_to_optimize_d["joint_record"]["val"]
            joint_loss_name = losses_to_optimize_d["joint_name"]

            # get metrics
            metric_collection = in_hash_to_metrics_config[
                make_hash(opt_instructs["in_conf"])
            ]
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
                # TODO: pass trackers here
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
                # add number of instances that have gone through the model for training
                total_train_inst += x_batch_train.shape[0]

                if all_train_step % LOGSTEPSIZE == 0:
                    log_model_params(tr_writer, all_train_step, model)
                    HIST_LOGGED = True

            logger.debug(f"END iterating training dataset- epoch: {e}")

            # TODO: add to tensorboard

            train_loss_updates = update_loss_trackers(
                "train",
                total_train_inst,
                loss_trackers,
                l2o_names,
                l2o_loss_record_train,
            )

            train_best_met_update = update_metric_trackers(
                "train",
                total_train_inst,
                metric_trackers,
                metric_names,
                metric_objs_train,
            )

            # TODO: adjust
            train_best_joint_update = record_joint_losses(
                "train",
                "epoch",
                e,
                joint_dict_tracker,
                joint_loss_name,
                joint_loss_record_train,
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

            train_loss_updates = update_loss_trackers(
                "val", total_train_inst, loss_trackers, l2o_names, l2o_loss_record_val
            )

            val_best_joint_update = record_joint_losses(
                "val",
                "epoch",
                e,
                joint_dict_tracker,
                joint_loss_name,
                joint_loss_record_val,
            )

            val_best_met_update = update_metric_trackers(
                "val", total_train_inst, metric_trackers, metric_names, metric_objs_val
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
        "loss": loss_trackers,
        "joint": joint_dict_tracker,
        "metrics": metric_trackers,
    }

    return return_dict
