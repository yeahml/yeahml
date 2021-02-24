import os
import pathlib
from typing import Any, Dict
from pathlib import Path

import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metric import configure_metric
from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.dataset.util import get_configured_dataset
from yeahml.log.yf_logging import config_logger
from yeahml.train.inference import inference_on_ds

#####
from yeahml.train.util import (
    convert_to_single_pass_iterator,
    get_losses_to_update,
    get_next_batch,
    re_init_iter,
)
from yeahml.train.gradients.gradients import get_validation_step_fn

################
from yeahml.config.create_configs import make_hash
from yeahml.train.setup.datasets import get_datasets
from yeahml.train.setup.paths import create_model_run_path
from yeahml.train.setup.objectives import get_objectives


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
    metric/losses by the same in_config (which may be the same prediction node
    and label), which is indexed by a hash.

    the chash_to_in_config, is then used to map the hash to which nodes to pull
    from on the graph


    e.g.
    `ds_to_chash`
        {
            "abalone": {
                -2976269282734729230: {"metric": set(), "loss": {"second_obj", "main_obj"}}
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
            if "metric" in cur_obj_dict.keys():
                if cur_obj_dict["metric"]:
                    metrics_objective_names.add(objective)

            # make the outter dict if not yet made
            try:
                _ = ds_to_chash[ds_name][conf_hash]
            except KeyError:
                ds_to_chash[ds_name][conf_hash] = {
                    "inference_fn": get_validation_step_fn()
                }

            try:
                s = ds_to_chash[ds_name][conf_hash]["metric"]
                if metrics_objective_names:
                    s.update(metrics_objective_names)
            except KeyError:
                ds_to_chash[ds_name][conf_hash]["metric"] = metrics_objective_names

            try:
                s = ds_to_chash[ds_name][conf_hash]["loss"]
                if loss_objective_names:
                    s.update(loss_objective_names)
            except KeyError:
                ds_to_chash[ds_name][conf_hash]["loss"] = loss_objective_names

    return ds_to_chash, chash_to_in_config


def create_output_index(model, chash_to_in_config):
    # TODO: this block should be abstracted away to be used by both
    # train/inference loops
    # TODO: this is hardcoded for supervised settings
    # tf.keras models output the model outputs in a list, we need to get the index
    # of each prediction we care about from that output to use in the loss
    # function
    # TODO: Additionally, this assumes the targeted variable is a model `output`
    # NOTE: rather than `None` when a model only outputs a single value, maybe
    # there should be a magic keyword/way to denote this.
    chash_to_output_index = {}
    if isinstance(model.output, list):
        MODEL_OUTPUT_ORDER = [n.name.split("/")[0] for n in model.output]
        for chash, cur_in_config in chash_to_in_config.items():
            try:
                assert (
                    cur_in_config["type"] == "supervised"
                ), f"only supervised is currently allowed, not {cur_in_config['type']} :("
                pred_name = cur_in_config["options"]["prediction"]
                out_index = MODEL_OUTPUT_ORDER.index(pred_name)
                chash_to_output_index[chash] = out_index
            except KeyError:
                # TODO: perform check later
                chash_to_output_index[chash] = None
    else:
        for chash, cur_in_config in chash_to_in_config.items():
            assert (
                cur_in_config["type"] == "supervised"
            ), f"only supervised is currently allowed, not {cur_in_config['type']} :("
            assert (
                model.output.name.split("/")[0]
                == cur_in_config["options"]["prediction"]
            ), f"model output {model.output.name.split('/')[0]} does not match prediction of cur_in_config: {cur_in_config}"
            chash_to_output_index[chash] = None

    return chash_to_output_index


def load_targeted_weights():
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
    raise NotImplementedError(f"not yet implemented")


def eval_model(
    model: Any,
    config_dict: Dict[str, Dict[str, Any]],
    datasets: Any = None,
    weights_path: str = "",
    eval_split="test",
    pred_dict=None,  # stupid hacky fix
) -> Dict[str, Any]:

    # TODO: allow for multiple splits to evaluate on

    # TODO: load the best weights
    # model = load_targeted_weights(full_exp_path, weights_path)

    # NOTE: should I reset the metrics?
    # # reset metrics (should already be reset)
    # for eval_metric_fn in eval_metric_fns:
    #     eval_metric_fn.reset_states()

    # # TODO: log each instance

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

    # create output index
    chash_to_output_index = create_output_index(model, chash_to_in_config)

    # objectives to objects
    # TODO: "test" should be obtained from the config
    # this returns a in_config, which isn't really needed.
    # TODO: is this always only going to be a single split?
    split_name = eval_split  # TODO: this needs to be double checked
    objectives_to_objects = get_objectives(
        perf_cdict["objectives"], dataset_dict, target_splits=split_name
    )

    logger.info("START - evaluating")
    ret_dict = {}
    for cur_ds_name, chash_conf_d in ds_to_chash.items():
        ret_dict[cur_ds_name] = {}
        logger.info(f"current dataset: {cur_ds_name}")

        for in_hash, cur_hash_conf in chash_conf_d.items():
            logger.info(f"in_hash: {in_hash}")
            ret_dict[cur_ds_name][in_hash] = {}
            cur_objective_config = chash_to_in_config[in_hash]
            assert (
                cur_objective_config["type"] == "supervised"
            ), f"only supervised is currently allowed, not {cur_objective_config['type']} :("
            logger.info(f"current config: {cur_objective_config}")

            cur_inference_fn = cur_hash_conf["inference_fn"]
            cur_metrics_objective_names = cur_hash_conf["metric"]
            cur_loss_objective_names = cur_hash_conf["loss"]
            cur_dataset_iter = dataset_iter_dict[cur_ds_name][split_name]
            cur_pred_index = chash_to_output_index[in_hash]
            cur_target_name = cur_objective_config["options"]["target"]

            temp_ret = inference_on_ds(
                model,
                cur_dataset_iter,
                cur_inference_fn,
                cur_loss_objective_names,
                cur_metrics_objective_names,
                objectives_to_objects,
                cur_pred_index,
                cur_target_name,
                eval_split,
                logger,
                pred_dict,
            )
            ret_dict[cur_ds_name][in_hash] = temp_ret

            # reinitialize validation iterator
            dataset_iter_dict[cur_ds_name][split_name] = re_init_iter(
                cur_ds_name, split_name, dataset_dict
            )

    return ret_dict
