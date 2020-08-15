import os
import pathlib
from typing import Any, Dict
from pathlib import Path

import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metric import configure_metric
from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.dataset.util import get_configured_dataset
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.train.inference import inference_dataset

#####
from yeahml.train.util import (
    convert_to_single_pass_iterator,
    get_losses_to_update,
    get_next_batch,
    re_init_iter,
)
from yeahml.train.gradients.gradients import get_validation_step_fn

################
from yeahml.config.model.util import make_hash
from yeahml.train.setup.datasets import get_datasets
from yeahml.train.setup.paths import create_model_run_path


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


def create_ds_to_lm_mapping(perf_cdict):
    """

    For evaluation, we would like to be efficient with looping over the
    datasets.
    
    ds_to_chash is used to iterate on the dataset on the outer loop then group
    metrics/losses by the same in_config (which may be the same prediction node
    and label), which is indexed by a hash.

    the chash_to_in_config, is then used to map the hash to which nodes to pull
    from on the graph


    e.g.
    `ds_to_chash`
        {
            "abalone": {
                -2976269282734729230: {"metrics": set(), "loss": {"second_obj", "main_obj"}}
            }
        }


    `chash_to_in_config`
        {
            -2976269282734729230: {
                "type": "supervised",
                "options": {"prediction": "dense_out", "target": "target_v"},
                "dataset": "abalone",
            }
        }
    
    """

    # map datasets to objectives
    ds_to_obj = {}
    for key, val in perf_cdict["objectives"].items():
        ds = val["in_config"]["dataset"]
        try:
            ds_to_obj[ds].append(key)
        except KeyError:
            ds_to_obj[ds] = [key]

    # map objectives to losses and metrics
    ds_to_chash = {}
    chash_to_in_config = {}
    for ds_name, ds_objs in ds_to_obj.items():
        ds_to_chash[ds_name] = {}
        loss_objective_names = set()
        metrics_objective_names = set()
        for objective in ds_objs:
            cur_obj_dict = perf_cdict["objectives"][objective]
            conf_hash = make_hash(cur_obj_dict["in_config"])
            try:
                assert (
                    chash_to_in_config[conf_hash] == cur_obj_dict["in_config"]
                ), f"hash not equal"
            except KeyError:
                chash_to_in_config[conf_hash] = cur_obj_dict["in_config"]
            if "loss" in cur_obj_dict.keys():
                if cur_obj_dict["loss"]:
                    loss_objective_names.add(objective)
            if "metrics" in cur_obj_dict.keys():
                if cur_obj_dict["metrics"]:
                    metrics_objective_names.add(objective)

            # make the outter dict if not yet made
            try:
                _ = ds_to_chash[ds_name][conf_hash]
            except KeyError:
                ds_to_chash[ds_name][conf_hash] = {
                    "inference_fn": get_validation_step_fn()
                }

            try:
                s = ds_to_chash[ds_name][conf_hash]["metrics"]
                if metrics_objective_names:
                    s.update(metrics_objective_names)
            except KeyError:
                ds_to_chash[ds_name][conf_hash]["metrics"] = metrics_objective_names

            try:
                s = ds_to_chash[ds_name][conf_hash]["loss"]
                if loss_objective_names:
                    s.update(loss_objective_names)
            except KeyError:
                ds_to_chash[ds_name][conf_hash]["loss"] = loss_objective_names

    return ds_to_chash, chash_to_in_config


def eval_model(
    model: Any,
    config_dict: Dict[str, Dict[str, Any]],
    datasets: Any = None,
    weights_path: str = "",
) -> Dict[str, Any]:

    # TODO: option to reinitialize model?

    # unpack configurations
    model_cdict: Dict[str, Any] = config_dict["model"]

    # set up loop for performing inference more efficiently
    perf_cdict: Dict[str, Any] = config_dict["performance"]
    ds_to_chash, chash_to_in_config = create_ds_to_lm_mapping(perf_cdict)

    # obtain datasets
    # TODO: hyperparams (depending on implementation) may not be relevant here
    data_cdict: Dict[str, Any] = config_dict["data"]
    hp_cdict: Dict[str, Any] = config_dict["hyper_parameters"]
    dataset_dict = get_datasets(datasets, data_cdict, hp_cdict)
    dataset_iter_dict = convert_to_single_pass_iterator(dataset_dict)

    # obtain logger
    log_cdict: Dict[str, Any] = config_dict["logging"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    full_exp_path = (
        pathlib.Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
        .joinpath(model_cdict["name"])
    )
    # build paths and obtain tb writers
    model_run_path = create_model_run_path(full_exp_path)
    logger = config_logger(model_run_path, log_cdict, "eval")

    print("====" * 8)
    print(ds_to_chash)
    print("----")
    print(chash_to_in_config)
    print("-----")
    print(dataset_iter_dict)

    raise NotImplementedError(f"not implemented yet")

    # # create a tf.function for applying gradients for each optimizer
    # # TODO: I am not 100% about this logic for maping the optimizer to the
    # #   apply_gradient fn... this needs to be confirmed to work as expected

    # opt_to_validation_fn = {}
    # opt_to_get_grads_fn, opt_to_app_grads_fn = {}, {}
    # opt_to_steps = {}
    # # used to determine which objectives to loop to calculate losses
    # opt_to_loss_objectives = {}
    # # used to determine which objectives to obtain to calculate metrics
    # opt_to_metrics_objectives = {}

    # # TODO: this is hardcoded for supervised settings
    # # tf.keras models output the model outputs in a list, we need to get the
    # # of each prediction we care about from that output to use in the loss
    # # function
    # # NOTE: I'm not sure how I feel about this -- is it better to have multiple
    # # "tf.models" that share params (is that even possible) -- or is it better
    # # to do this where it is one "tf.model"?
    # if isinstance(model.output, list):
    #     MODEL_OUTPUT_ORDER = [n.name.split("/")[0] for n in model.output]
    #     objective_to_output_index = {}
    #     for obj_name, obj_dict in objectives_dict.items():
    #         try:
    #             pred_name = obj_dict["in_config"]["options"]["prediction"]
    #             out_index = MODEL_OUTPUT_ORDER.index(pred_name)
    #             objective_to_output_index[obj_name] = out_index
    #         except KeyError:
    #             # TODO: perform check later
    #             objective_to_output_index[obj_name] = None
    # else:
    #     # TODO: this is hardcoded to assume supervised
    #     objective_to_output_index = {}
    #     for obj_name, obj_dict in objectives_dict.items():
    #         objective_to_output_index[obj_name] = None

    # logger.info("START - evaluating")

    # # loss
    # # opt_name :loss :main_obj :ds_name :split_name :loss_name:desc_name
    # # opt_name :metric :main_obj: ds_name :split_name :metric_name
    # opt_tracker_dict = main_tracker_dict[cur_optimizer_name]

    # # TODO: these should really be grouped by the in config (likely by
    # # creating a hash) this allows us to group objectives by what
    # # dataset their using so that we can reuse the same batch.
    # # NOTE: for now, I'm saving the prediction and gt (if supervised) in
    # # the grad_dict
    # loss_objective_names = opt_to_loss_objectives[cur_optimizer_name]
    # metrics_objective_names = opt_to_metrics_objectives[cur_optimizer_name]

    # obj_to_grads = {}
    # # TODO: the losses should be grouped by the ds used so that we only
    # # obtain+run the batch once+ensuring it's the same batch
    # loss_update_dict, update_metrics_dict = {}, {}

    # cur_objective = select_objective(loss_objective_names)
    # logger.info(f"objective: {cur_objective}")
    # continue_objective = True

    # # TODO: next step -- continue_objective = True
    # # each loss may be being optimized by data from different datasets
    # cur_ds_name = objectives_dict[cur_objective]["in_config"]["dataset"]
    # loss_conf = objectives_dict[cur_objective]["loss"]
    # tf_train_loss_descs_to_update = get_losses_to_update(loss_conf, "train")

    # cur_train_iter = get_train_iter(dataset_iter_dict, cur_ds_name, "train")

    # while continue_objective:
    #     cur_batch = get_next_batch(cur_train_iter)
    #     if not cur_batch:

    #         # dataset pass is complete
    #         obj_ds_to_epoch = update_epoch_dict(
    #             obj_ds_to_epoch, cur_objective, cur_ds_name, "train"
    #         )

    #         if (
    #             obj_ds_to_epoch[cur_objective][cur_ds_name]["train"]
    #             >= hp_cdict["epochs"]
    #         ):

    #             # update this particular combination to false -
    #             # eventually this logic will be "smarter" i.e. not
    #             # based entirely on number of epochs.
    #             opt_obj_ds_to_training[cur_optimizer_name][cur_objective][cur_ds_name][
    #                 "train"
    #             ] = False

    #             # this objective is done. see if they're all done
    #             is_training = determine_if_training(opt_obj_ds_to_training)

    #             # TODO: this isn't the "best" way to handle this,
    #             # ideally, we would decided (in an intelligent way) when
    #             # we're done training a group of objectives by
    #             # evaluating the loss curves
    #             list_of_optimizers.remove(cur_optimizer_name)
    #             logger.info(
    #                 f"{cur_optimizer_name} removed from list of opt. remaining: {list_of_optimizers}"
    #             )
    #             logger.info(f"is_training: {is_training}")
    #             # TODO: determine whether to move to the next objective
    #             # NOTE: currently, move to the next objective

    #             # TODO: there is likely a better way to handle the case
    #             # where we have reached the 'set' number of epochs for
    #             # this problem

    #         # the original dict is updated here in case another dataset
    #         # needs to use the datset iter -- this could likely be
    #         # optimized, but the impact would be minimal right now
    #         cur_train_iter = re_init_iter(cur_ds_name, "train", dataset_dict)
    #         dataset_iter_dict[cur_ds_name]["train"] = cur_train_iter

    #         logger.info(
    #             f"epoch {cur_objective} - {cur_ds_name} {'train'}:"
    #             f" {obj_ds_to_epoch[cur_objective][cur_ds_name]['train']}"
    #         )

    #         # perform validation after each pass through the training
    #         # dataset
    #         # NOTE: the location of this 'validation' may change
    #         # TODO: there is an error here where the first objective
    #         # will be validated on the last epoch and then one more
    #         # time.
    #         # TODO: ensure the metrics are reset
    #         #  iterate validation after iterating entire training..
    #         # this will/should change to update on a set frequency --
    #         # also, maybe we don't want to run the "full" validation,
    #         # only a (random) subset?

    #         # validation pass
    #         cur_val_update = inference_dataset(
    #             model,
    #             loss_objective_names,
    #             metrics_objective_names,
    #             dataset_iter_dict,
    #             opt_to_validation_fn[cur_optimizer_name],
    #             opt_tracker_dict,
    #             cur_objective,
    #             cur_ds_name,
    #             dataset_dict,
    #             opt_to_steps[cur_optimizer_name],
    #             num_training_ops,
    #             objective_to_output_index,
    #             objectives_dict,
    #             v_writer,
    #             logger,
    #             split_name="val",
    #         )

    #     else:

    #         # create histograms of model parameters
    #         if log_cdict["track"]["tensorboard"]["param_steps"] > 0:
    #             if (
    #                 num_training_ops % log_cdict["track"]["tensorboard"]["param_steps"]
    #                 == 0
    #             ):
    #                 log_model_params(tr_writer, num_training_ops, model)

    #         # update Tracker
    #         if log_cdict["track"]["tracker_steps"] > 0:
    #             if num_training_ops % log_cdict["track"]["tracker_steps"] == 0:
    #                 cur_loss_tracker_dict = opt_tracker_dict[cur_objective]["loss"][
    #                     cur_ds_name
    #                 ]["train"]
    #                 cur_loss_update = update_loss_trackers(
    #                     loss_conf["track"]["train"],
    #                     cur_loss_tracker_dict,
    #                     opt_to_steps[cur_optimizer_name],
    #                     num_training_ops,
    #                     tb_writer=tr_writer,
    #                     ds_name=cur_ds_name,
    #                     objective_name=cur_objective,
    #                 )

    #                 loss_update_dict[cur_objective] = cur_loss_update

    #         # TODO: this is a hacky way of seeing if training on a batch was run

    #     update_dict = {"loss": loss_update_dict, "metrics": update_metrics_dict}
    # # one pass of training (a batch from each objective) with the
    # # current optimizer

    # # TODO: I think the 'joint' should likely be the optimizer name, not the
    # # combination of losses name, this would also simplify the creation of these

    # return_dict = {"tracker": main_tracker_dict}

    # return return_dict

    # raise NotImplementedError(
    #     "this functionality is currently broken and needs to be updated"
    # )

    # # TODO: it also be possible to use this without passing the model?

    # # load best weights
    # # TODO: load specific weights according to a param

    # model_path = full_exp_path.joinpath("model")
    # if model_path.is_dir():
    #     sub_dirs = [x for x in model_path.iterdir() if x.is_dir()]
    #     most_recent_subdir = sub_dirs[0]
    #     # TODO: this assumes .h5 is the filetype and that model.h5 is present
    #     most_recent_save_path = Path(most_recent_subdir).joinpath("save")

    # # TODO: THis logic/specification needs to be validated. Is this what we
    # # want? what if we want to specify a specific run? Where should we get that information?
    # if weights_path:
    #     specified_path = weights_path
    # else:
    #     # TODO: this needs to allow an easier specification for which params to load
    #     # The issue here is that if someone alters the model (perhaps in a notebook), then
    #     # retrains the model (but doesn't update the config), the "old" model, not new model
    #     # will be evaluated.  This will need to change
    #     specified_path = most_recent_save_path.joinpath("params").joinpath(
    #         "best_params.h5"
    #     )

    # if not specified_path.is_file():
    #     raise ValueError(
    #         f"specified path is neither an h5 path, nor a directory containing directories of h5: {specified_path}"
    #     )

    # model.load_weights(str(specified_path))
    # logger.info(f"params loaded from {specified_path}")

    # # Right now, we're only going to add the first loss to the existing train
    # # loop
    # objective_list = list(perf_cdict["objectives"].keys())
    # if len(objective_list) > 1:
    #     raise ValueError(
    #         "Currently, only one objective is supported by the training loop logic. There are {len(objective_list)} specified ({objective_list})"
    #     )
    # first_and_only_obj = objective_list[0]

    # # loss
    # # get loss function
    # loss_object = configure_loss(perf_cdict["objectives"][first_and_only_obj]["loss"])

    # # mean loss
    # avg_eval_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)

    # # metrics
    # eval_metric_fns, metric_order = [], []
    # # TODO: this is hardcoded to only the first objective
    # met_opts = perf_cdict["objectives"][first_and_only_obj]["metric"]["options"]
    # # TODO: this is hardcoded to only the first objective
    # for i, metric in enumerate(
    #     perf_cdict["objectives"][first_and_only_obj]["metric"]["type"]
    # ):
    #     try:
    #         met_opt_dict = met_opts[i]
    #     except TypeError:
    #         # no options
    #         met_opt_dict = None
    #     except IndexError:
    #         # No options for particular metric
    #         met_opt_dict = None
    #     eval_metric_fn = configure_metric(metric, met_opt_dict)
    #     eval_metric_fns.append(eval_metric_fn)
    #     metric_order.append(metric)

    # # reset metrics (should already be reset)
    # for eval_metric_fn in eval_metric_fns:
    #     eval_metric_fn.reset_states()

    # # get datasets
    # # TODO: ensure eval dataset is the same as used previously
    # # TODO: apply shuffle/aug/reshape from config
    # if not dataset:
    #     eval_ds = get_configured_dataset(data_cdict, hp_cdict)
    # else:
    #     assert isinstance(
    #         dataset, tf.data.Dataset
    #     ), f"a {type(dataset)} was passed as a test dataset, please pass an instance of {tf.data.Dataset}"
    #     eval_ds = get_configured_dataset(data_cdict, hp_cdict, ds=dataset)

    # logger.info("-> START evaluating model")
    # for step, (x_batch, y_batch) in enumerate(eval_ds):
    #     eval_step(model, x_batch, y_batch, loss_object, avg_eval_loss, eval_metric_fns)
    # logger.info("[END] evaluating model")

    # logger.info("-> START creating eval_dict")
    # eval_dict = {}
    # for i, name in enumerate(metric_order):
    #     cur_metric_fn = eval_metric_fns[i]
    #     eval_dict[name] = cur_metric_fn.result().numpy()
    # logger.info("[END] creating eval_dict")

    # # TODO: log each instance

    # return eval_dict
