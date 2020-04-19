import pathlib
from typing import Any, Dict

import tensorflow as tf

from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.train.setup.datasets import get_datasets
from yeahml.train.setup.objectives import get_objectives
from yeahml.train.setup.paths import (
    create_model_run_path,
    create_model_training_paths,
    get_tb_writers,
)
from yeahml.train.setup.tracker.metric import (
    create_metric_trackers,
    update_metric_trackers,
)
from yeahml.train.setup.tracker.tracker import (
    create_joint_dict_tracker,
    record_joint_losses,
)

from yeahml.train.setup.loop_dynamics import (  # obtain_optimizer_loss_mapping,; create_grouped_metrics,; map_in_config_to_objective,
    create_full_dict,
    get_optimizers,
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

- could keep note of all extra tricky batches/instances after n epochs.
"""


"""
TODO: check that all metrics are accounted for.  If so. raise a not
implemented error -- presently the training loop is driven by the
optimizers (and as a result all objectives that have matching in_configs).
meaning, if a metric does not have a matching in_config, it will not be
evaluated.

# TODO: random -- check for nans in loss values

TODO: build best loss dict

TODO: ASSUMPTION: using optimizers sequentially. this may be:
- jointly, ordered: sequentially, or unordered: alternate/random

NOTE: I'm not sure looping on epochs makes sense as an outter layer
anymore.
TODO: is there a way to save a variable in the graph to keep track of
epochs (if multiple runs from a notebook?)

TODO: per batch/epoch/adaptive -- all open questions.

NOTE: this isn't perfect in that someone may care about the output of
something they are not directly optimizing.. I'd argue the current
workaround is to specify that metric with a dataset and associate it to a
particular optimizer (it will eval at the same time this optimizer is run)


"""


# TODO: this is supervised
def get_get_supervised_grads_fn():

    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    @tf.function
    def get_grad(model, batch, loss_fns):
        # supervised implies a x, and y.. however, this maybe should change to a
        # dict indexing
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]

        x_batch, y_batch = batch
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

        return {
            "gradients": grads,
            "predictions": prediction,
            "final_loss": final_loss,
            "losses": loss,
            "y_batch": y_batch,
        }
        # return grads, prediction, final_loss, full_losses

    return get_grad


def get_apply_grad_fn():

    # https://github.com/tensorflow/tensorflow/issues/27120
    # this allows the model to continue to be trained on multiple calls
    @tf.function
    def apply_grad(model, grads, optimizer):

        # NOTE: any gradient adjustments would happen here
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return

    return apply_grad


def get_validation_step_fn():
    @tf.function
    def get_preds(model, batch, loss_fns):
        # supervised implies a x, and y.. however, this maybe should change to a
        # dict indexing
        if not isinstance(loss_fns, list):
            loss_fns = [loss_fns]

        x_batch, y_batch = batch
        prediction = model(x_batch, training=False)

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

        return {
            "predictions": prediction,
            "final_loss": final_loss,
            "losses": loss,
            "y_batch": y_batch,
        }

    return get_preds


def log_model_params(tr_writer, g_train_step, model):
    with tr_writer.as_default():
        for v in model.variables:
            tf.summary.histogram(v.name.split(":")[0], v.numpy(), step=g_train_step)


def get_next_batch(ds_iter):

    # TODO: this should accept the ds_dict and ds_iter so that if we reach the
    # we can recreate the ds_iter -- this will allow us to keep track of number
    # of passes for each dataset. When we do this, we'll also have to drop the
    # `convert_to_single_pass_iterator` function --- it will be included here
    # instead.
    try:
        batch = next(ds_iter)
    except StopIteration:
        batch = None
    return batch


def convert_to_iter(tf_ds):
    return tf_ds.repeat(1).__iter__()


def re_init_iter(ds_name, split_name, ds_dict):
    return convert_to_iter(ds_dict[ds_name][split_name])


def convert_to_single_pass_iterator(ds_dict):
    iter_dict = {}
    for ds_name, ds_name_conf in ds_dict.items():
        iter_dict[ds_name] = {}
        for split_name, tf_ds in ds_name_conf.items():
            # only loop once
            iter_dict[ds_name][split_name] = convert_to_iter(tf_ds)
    return iter_dict


def update_metrics_tracking(
    metrics_objective_names,
    objectives_dict,
    opt_tracker_dict,
    obj_to_grads,
    num_train_instances,
    ds_split_name,
):
    # update Tracker and reset tf object
    # NOTE: presently there can only be one loss coming in so the order is not
    # important

    # TODO: need to ensure (outside this function) that the predictions are the
    # same shape as the y_batch such that we don't have a broadcasting issue

    # I'm not 100% sure the indexing of losses here...

    # update the tf description (e.g. mean) as well as the Tracker

    update_dict = {}

    for cur_objective in metrics_objective_names:
        update_dict[cur_objective] = {}
        cur_ds_name = objectives_dict[cur_objective]["in_config"]["dataset"]
        cur_metric_tracker_dict = opt_tracker_dict[cur_objective]["metrics"][
            cur_ds_name
        ][ds_split_name]

        metric_conf = objectives_dict[cur_objective]["metrics"]
        for metric_name, split_to_metric in metric_conf.items():
            metric_tracker = cur_metric_tracker_dict[metric_name]
            update_dict[cur_objective][metric_name] = {}
            if ds_split_name in split_to_metric.keys():
                metric_obj = split_to_metric[ds_split_name]
                result = metric_obj.result().numpy()
                metric_obj.reset_states()
                cur_update = metric_tracker.update(
                    step=num_train_instances, value=result
                )
                update_dict[cur_objective][metric_name] = cur_update

    return update_dict


def update_epoch_dict(
    obj_ds_to_epoch: Dict[str, Any],
    objective_name: str,
    dataset_name: str,
    split_name: str,
):
    """update num of iterations
    """
    try:
        obj_ds_to_epoch[objective_name][dataset_name][split_name] += 1
    except KeyError:
        tmp_dict = {objective_name: {dataset_name: {split_name: 0}}}
        obj_ds_to_epoch = {**obj_ds_to_epoch, **tmp_dict}
    return obj_ds_to_epoch


def update_is_training_dict(
    obj_ds_to_training, objective_name, dataset_name, split_name
):
    try:
        obj_ds_to_training[objective_name][dataset_name][split_name] = False
    except KeyError:
        obj_ds_to_training = {objective_name: {dataset_name: {split_name: True}}}


def determine_if_training(obj_ds_to_training):

    # if a single is_training is found return True, else they are all false,
    # return false
    for obj_name, obj_to_training in obj_ds_to_training.items():
        for ds_name, ds_to_training in obj_to_training.items():
            for split_name, is_training in ds_to_training.items():
                if is_training:
                    return True
    else:
        return False


def combine_gradients(obj_to_grads):
    # TODO: need to research how best to combine the gradients here...
    # combine all gradients. This portion (with in the optimizer loop)
    # will combine the gradients as if it were trained jointly
    combined_gradients = None
    for obj_name, grad_dict in obj_to_grads.items():
        # TODO: we could add scaling/weighting here
        if not combined_gradients:
            combined_gradients = grad_dict["gradients"]
        else:
            combined_gradients += grad_dict["gradients"]
    return combined_gradients


def update_model_params(apply_grads_fn, obj_to_grads, model, cur_tf_optimizer):
    # combine gradients and use optimizer to update model. apply contraints to
    # model variables
    combined_gradients = combine_gradients(obj_to_grads)

    # apply gradients to the model
    apply_grads_fn(model, combined_gradients, cur_tf_optimizer)

    # apply constraints
    for variable in model.variables:
        if variable.constraint is not None:
            variable.assign(variable.constraint(variable))


def update_metric_objects(
    metrics_objective_names, objectives_dict, obj_to_grads, split_name
):

    # NOTE: do we need to combine predictions here at all?

    for cur_objective in metrics_objective_names:
        # could make hash of this
        cur_in_conf = objectives_dict[cur_objective]["in_config"]
        # {
        #     "type": "supervised",
        #     "options": {"prediction": "dense_out", "target": "target_v"},
        #     "dataset": "abalone",
        # }
        # cur_ds_name = cur_in_conf["dataset"]
        metric_conf = objectives_dict[cur_objective]["metrics"]
        # {'meanabsoluteerror': {'train': "tf.metric", 'val':
        # "tf.metric"}}
        for metric_name, split_to_metric in metric_conf.items():
            if split_name in split_to_metric.keys():
                metric_obj = split_to_metric[split_name]

                # TODO: hardcoded
                if cur_in_conf["type"] == "supervised":
                    preds = obj_to_grads[cur_objective]["predictions"]
                    y_batch = obj_to_grads[cur_objective]["y_batch"]
                    metric_obj.update_state(y_batch, preds)


def update_tf_val_losses(pred_dict, track_desc_dict):
    # the state is not reset here
    # TODO: need to ensure (outside this function) that the predictions are the
    # same shape as the y_batch such that we don't have a broadcasting issue
    for _, desc_dict in track_desc_dict.items():
        #  _ = loss_name

        for desc_name, desc_tf_obj in desc_dict.items():
            losses = pred_dict["losses"]
            desc_tf_obj.update_state(losses)


def update_val_loss_trackers(cur_loss_conf, cur_loss_tracker_dict, num_train_instances):
    # update Tracker and reset tf states

    update_dict = {}
    for loss_name, desc_dict in cur_loss_conf.items():
        update_dict[loss_name] = {}
        for desc_name, desc_tf_obj in desc_dict.items():
            desc_tracker = cur_loss_tracker_dict[loss_name][desc_name]

            tf_desc_val = desc_tf_obj.result().numpy()
            desc_tf_obj.reset_states()

            cur_update = desc_tracker.update(
                step=num_train_instances, value=tf_desc_val
            )
            update_dict[loss_name][desc_name] = cur_update

    return update_dict


def update_tf_val_metrics(val_preds_dict, metrics_conf, val_name, cur_metrics_type):

    for metric_name, split_to_metric in metrics_conf.items():
        if val_name in split_to_metric.keys():
            metric_tf_obj = split_to_metric[val_name]

            # TODO: hardcoded - some may not be a prediction/ground truth
            if cur_metrics_type == "supervised":
                preds = val_preds_dict["predictions"]
                y_batch = val_preds_dict["y_batch"]
                metric_tf_obj.update_state(y_batch, preds)


def update_val_metrics_trackers(
    metrics_conf, cur_metric_tracker_dict, val_name, num_train_instances
):
    # update Tracker, reset tf states
    update_dict = {}

    for metric_name, split_to_metric in metrics_conf.items():
        metric_tracker = cur_metric_tracker_dict[metric_name]
        update_dict[metric_name] = {}
        if val_name in split_to_metric.keys():
            metric_obj = split_to_metric[val_name]
            result = metric_obj.result().numpy()
            metric_obj.reset_states()
            cur_update = metric_tracker.update(step=num_train_instances, value=result)
            update_dict[metric_name] = cur_update

    return update_dict


def validation(
    model,
    loss_objective_names,
    metrics_objective_names,
    dataset_iter_dict,
    cur_val_fn,
    opt_tracker_dict,
    cur_objective,
    cur_ds_name,
    dataset_dict,
    num_train_instances,
    objectives_dict,
):
    val_name = "val"

    cur_update = {}
    for cur_objective in loss_objective_names:
        cur_in_conf = objectives_dict[cur_objective]["in_config"]
        cur_ds_name = cur_in_conf["dataset"]
        cur_ds_iter_dict = dataset_iter_dict[cur_ds_name]
        if val_name not in cur_ds_iter_dict.keys():
            raise ValueError(
                f"{cur_in_conf['dataset']} does not have a '{val_name}' dataset"
            )
        cur_val_iter = cur_ds_iter_dict[val_name]

        # loss
        loss_conf = objectives_dict[cur_objective]["loss"]
        cur_loss_conf = loss_conf["track"][val_name]

        # metrics
        metrics_conf = objectives_dict[cur_objective]["metrics"]

        # iterate batches
        cur_batch = get_next_batch(cur_val_iter)

        while cur_batch:

            val_dict = cur_val_fn(model, cur_batch, loss_conf["object"])
            # {"predictions": prediction,
            # "final_loss": final_loss,
            # "losses": loss,
            # "y_batch": y_batch,}

            # update tf objects
            update_tf_val_losses(val_dict, cur_loss_conf)
            update_tf_val_metrics(val_dict, metrics_conf, val_name, cur_in_conf["type"])

            # next batch until end
            cur_batch = get_next_batch(cur_val_iter)

        # reinitialize validation iterator
        dataset_iter_dict[cur_ds_name][val_name] = re_init_iter(
            cur_ds_name, val_name, dataset_dict
        )

        # update trackers
        cur_loss_tracker_dict = opt_tracker_dict[cur_objective]["loss"][cur_ds_name][
            val_name
        ]
        cur_loss_update = update_val_loss_trackers(
            cur_loss_conf, cur_loss_tracker_dict, num_train_instances
        )

        cur_metric_tracker_dict = opt_tracker_dict[cur_objective]["metrics"][
            cur_ds_name
        ][val_name]
        cur_metrics_update = update_val_metrics_trackers(
            metrics_conf, cur_metric_tracker_dict, val_name, num_train_instances
        )

        cur_update[cur_objective] = {
            "loss": cur_loss_update,
            "metrics": cur_metrics_update,
        }
    return cur_update


def train_model(
    model: Any, config_dict: Dict[str, Dict[str, Any]], datasets: dict = None
) -> Dict[str, Any]:

    # TODO: option to reinitialize model?

    YML_TRACK_UPDATE = 30

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
    # train_ds, val_ds = get_datasets(datasets, data_cdict, hp_cdict)
    dataset_dict = get_datasets(datasets, data_cdict, hp_cdict)

    # {optimizer_name: {"optimizer": tf.obj, "objective": [objective_name]}}
    optimizers_dict = get_optimizers(optim_cdict)

    # {objective_name: "in_config": {...}, "loss": {...}, "metric": {...}}
    objectives_dict = get_objectives(perf_cdict["objectives"], dataset_dict)

    # create a tf.function for applying gradients for each optimizer
    # TODO: I am not 100% about this logic for maping the optimizer to the
    #   apply_gradient fn... this needs to be confirmed to work as expected
    # opt_name_to_gradient_fn = {}
    # get_apply_grad_fn
    # get_get_supervised_grads_fn
    opt_to_validation_fn = {}
    opt_to_get_grads_fn = {}
    opt_to_app_grads_fn = {}
    opt_to_steps = {}
    opt_to_val_runs = {}
    for cur_optimizer_name, _ in optimizers_dict.items():
        # opt_name_to_gradient_fn[cur_optimizer_name] = get_apply_grad_fn()
        # TODO: check config to see which fn to get supervised/etc
        opt_to_get_grads_fn[cur_optimizer_name] = get_get_supervised_grads_fn()
        opt_to_app_grads_fn[cur_optimizer_name] = get_apply_grad_fn()
        opt_to_validation_fn[cur_optimizer_name] = get_validation_step_fn()
        opt_to_steps[cur_optimizer_name] = 0
        opt_to_val_runs[cur_optimizer_name] = 1

    # TODO: training_directive may be empty.
    # {
    #     "YEAHML_1": {"optimizers": ["YEAHML_0", "second_opt"], "operation": "&"},
    #     "YEAHML_0": {"optimizers": ["main_opt", "second_opt"], "operation": ","},
    # }
    # TODO: I will need to parse this to create a cleaner directive to follow
    # training_directive = optim_cdict["directive"]

    main_tracker_dict = create_full_dict(
        optimizers_dict=optimizers_dict,
        objectives_dict=objectives_dict,
        datasets_dict=dataset_dict,
    )

    dataset_iter_dict = convert_to_single_pass_iterator(dataset_dict)

    # TODO: create list order of directives to loop through
    logger.debug("START - iterating epochs dataset")
    # all_train_step = 0
    # LOGSTEPSIZE = 10

    # TODO: how do I determine how "long" to go here... I think the 'right'
    # answer is dependent on the losses (train and val), but I think there is a
    # short answer as well.

    # TODO: this needs to be driven by the directive, not just a walkthrough
    obj_ds_to_epoch = {}
    obj_ds_to_training = {}
    # initialize to True
    is_training = True
    num_training_ops = 0
    while is_training:
        for cur_optimizer_name, cur_optimizer_config in optimizers_dict.items():
            # TODO: currently a single optimizer is run and then the other is run..
            # this is not really the way we'd like to approach this.

            # loss
            # opt_name :loss :main_obj :ds_name :split_name :loss_name:desc_name
            # opt_name :metric :main_obj: ds_name :split_name :metric_name
            opt_tracker_dict = main_tracker_dict[cur_optimizer_name]

            # NOTE: if there are multiple objectives, they will be trained *jointly*
            # cur_optimizer_config:
            #   {'optimizer': <tf.opt{}>, 'objectives': ['main_obj']}
            # cur_apply_grad_fn = opt_name_to_gradient_fn[cur_optimizer_name]
            get_grads_fn = opt_to_get_grads_fn[cur_optimizer_name]
            apply_grads_fn = opt_to_app_grads_fn[cur_optimizer_name]

            HIST_LOGGED = False  # will update for each optimizer
            logger.debug(f"START - optimizing {cur_optimizer_name}")

            # get optimizer
            cur_tf_optimizer = cur_optimizer_config["optimizer"]
            objectives_to_opt = cur_optimizer_config["objectives"]

            # TODO: should this happen outside the loop? I feel like yes..
            # TODO: these should really be grouped by the in config (likely by
            # creating a hash) this allows us to group objectives by what
            # dataset their using so that we can reuse the same batch.
            # NOTE: for now, I'm saving the prediction and gt (if supervised) in
            # the grad_dict
            # gather losses and metrics
            loss_objective_names = []
            metrics_objective_names = []
            for cur_objective in objectives_to_opt:
                cur_objective_dict = objectives_dict[cur_objective]
                if "loss" in cur_objective_dict.keys():
                    loss_objective_names.append(cur_objective)
                if "metrics" in cur_objective_dict.keys():
                    metrics_objective_names.append(cur_objective)

            # TODO: reset losses

            obj_to_grads = {}
            # TODO: the losses should be grouped by the ds used so that we only
            # obtain+run the batch once+ensuring it's the same batch
            loss_update_dict = {}
            for cur_objective in loss_objective_names:
                cur_in_conf = objectives_dict[cur_objective]["in_config"]
                loss_conf = objectives_dict[cur_objective]["loss"]

                cur_ds_name = cur_in_conf["dataset"]
                cur_ds_iter_dict = dataset_iter_dict[cur_ds_name]
                if "train" not in cur_ds_iter_dict.keys():
                    raise ValueError(
                        f"{cur_in_conf['dataset']} does not have a 'train' dataset"
                    )
                cur_train_iter = cur_ds_iter_dict["train"]

                # NOTE: := ?
                cur_batch = get_next_batch(cur_train_iter)
                if not cur_batch:
                    # have reached the end of the dataset
                    obj_ds_to_epoch = update_epoch_dict(
                        obj_ds_to_epoch, cur_objective, cur_ds_name, "train"
                    )
                    update_is_training_dict(
                        obj_ds_to_training, cur_objective, cur_ds_name, "train"
                    )
                    logger.debug(
                        f"epoch {cur_objective} - {cur_ds_name} {'train'}: {obj_ds_to_epoch[cur_objective][cur_ds_name]['train']}"
                    )
                    if (
                        obj_ds_to_epoch[cur_objective][cur_ds_name]["train"]
                        >= hp_cdict["epochs"]
                    ):
                        is_training = determine_if_training(obj_ds_to_training)
                        # TODO: there is likely a better way to handle the case
                        # where we have reached the 'set' number of epochs for
                        # this problem

                    dataset_iter_dict[cur_ds_name]["train"] = re_init_iter(
                        cur_ds_name, "train", dataset_dict
                    )
                    # print(f"e: {obj_ds_to_epoch[cur_objective][cur_ds_name]['train']}")

                    # perform validation after each pass through the training
                    # dataset
                    # NOTE: the location of this 'validation' may change
                    # TODO: there is an error here where the first objective
                    # will be validated on the last epoch and then one more
                    # time.
                    # TODO: ensure the metrics are reset
                    cur_val_fn = opt_to_validation_fn[cur_optimizer_name]
                    cur_val_update = validation(
                        model,
                        loss_objective_names,
                        metrics_objective_names,
                        dataset_iter_dict,
                        cur_val_fn,
                        opt_tracker_dict,
                        cur_objective,
                        cur_ds_name,
                        dataset_dict,
                        opt_to_steps[cur_optimizer_name],
                        objectives_dict,
                    )

                    break

                grad_dict = get_grads_fn(model, cur_batch, loss_conf["object"])

                # TODO: see note above about ensuring the same batch is used for
                # losses with the same dataset specified
                opt_to_steps[cur_optimizer_name] += cur_batch[0].shape[0]
                num_training_ops += 1
                # grad_dict contains {
                #     "gradients": grads,
                #     "predictions": prediction,
                #     "final_loss": final_loss,
                #     "losses": loss,
                # }
                obj_to_grads[cur_objective] = grad_dict

                # update Tracker

                cur_loss_tracker_dict = opt_tracker_dict[cur_objective]["loss"][
                    cur_ds_name
                ]["train"]
                cur_loss_conf_desc = loss_conf["track"]["train"]

                # NOTE: the steps here aren't accurate (due to note above about)
                # using the same batches for objectives/losses that specify the
                # same datasets
                # TODO: HERE
                update_tf_val_losses(grad_dict, cur_loss_conf_desc)

                if num_training_ops % YML_TRACK_UPDATE == 0:
                    cur_loss_update = update_val_loss_trackers(
                        cur_loss_conf_desc,
                        cur_loss_tracker_dict,
                        opt_to_steps[cur_optimizer_name],
                    )

                    loss_update_dict[cur_objective] = cur_loss_update

            # TODO: this is a hacky way of seeing if training on a batch was run
            update_metrics_dict = None
            if obj_to_grads:
                update_model_params(
                    apply_grads_fn, obj_to_grads, model, cur_tf_optimizer
                )

                update_metric_objects(
                    metrics_objective_names, objectives_dict, obj_to_grads, "train"
                )

                if num_training_ops % YML_TRACK_UPDATE == 0:
                    update_metrics_dict = update_metrics_tracking(
                        metrics_objective_names,
                        objectives_dict,
                        opt_tracker_dict,
                        obj_to_grads,
                        opt_to_steps[cur_optimizer_name],
                        "train",
                    )

            update_dict = {"loss": loss_update_dict, "metrics": update_metrics_dict}

            # one pass of training (a batch from each objective) with the
            # current optimizer

    # TODO: I think the 'joint' should likely be the optimizer name, not the
    # combination of losses name, this would also simplify the creation of these

    return_dict = {"tracker": main_tracker_dict}

    return return_dict

    # # TODO: add to tensorboard
    # if all_train_step % LOGSTEPSIZE == 0:
    #     log_model_params(tr_writer, all_train_step, model)
    #     HIST_LOGGED = True
    # logger.debug(f"END iterating training dataset- epoch: {e}")

    # # TODO: adjust
    # train_best_joint_update = record_joint_losses(
    #     "train",
    #     "epoch",
    #     e,
    #     joint_dict_tracker,
    #     joint_loss_name,
    #     joint_loss_record_train,
    # )

    # TODO: tensorboard
    # with tr_writer.as_default():
    #     tf.summary.scalar("loss", cur_train_loss_, step=e)
    #     for i, name in enumerate(metric_order):
    #         cur_train_metric_fn = train_metric_fns[i]
    #         tf.summary.scalar(name, cur_train_metric_fn.result().numpy(), step=e)

    # # This may not be the place to log these...
    # if not HIST_LOGGED:
    #     log_model_params(tr_writer, all_train_step, model)
    #     HIST_LOGGED = True

    # logger.debug(f"START iterating validation dataset - epoch: {e}")
    # # iterate validation after iterating entire training.. this will/should
    # # change to update on a set frequency -- also, maybe we don't want to
    # # run the "full" validation, only a (random) subset?

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

    # TODO: tensorboard
    # with v_writer.as_default():
    #     tf.summary.scalar("loss", cur_val_loss_, step=e)
    #     for i, name in enumerate(metric_order):
    #         cur_val_metric_fn = val_metric_fns[i]
    #         tf.summary.scalar(name, cur_val_metric_fn.result().numpy(), step=e)
