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
    config_dict: Dict[str, Dict[str, Any]],
    dataset: Any = None,
    weights_path: str = "",
) -> Dict[str, Any]:
    raise NotImplementedError(
        "this functionality is currently broken and needs to be updated"
    )

    # TODO: it also be possible to use this without passing the model?

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]
    hp_cdict: Dict[str, Any] = config_dict["hyper_parameters"]
    perf_cdict: Dict[str, Any] = config_dict["performance"]

    full_exp_path = (
        Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
    )
    logger = config_logger(full_exp_path, log_cdict, "eval")

    # load best weights
    # TODO: load specific weights according to a param

    model_path = full_exp_path.joinpath("model")
    if model_path.is_dir():
        sub_dirs = [x for x in model_path.iterdir() if x.is_dir()]
        most_recent_subdir = sub_dirs[0]
        # TODO: this assumes .h5 is the filetype and that model.h5 is present
        most_recent_save_path = Path(most_recent_subdir).joinpath("save")

    # TODO: THis logic/specification needs to be validated. Is this what we
    # want? what if we want to specify a specific run? Where should we get that information?
    if weights_path:
        specified_path = weights_path
    else:
        # TODO: this needs to allow an easier specification for which params to load
        # The issue here is that if someone alters the model (perhaps in a notebook), then
        # retrains the model (but doesn't update the config), the "old" model, not new model
        # will be evaluated.  This will need to change
        specified_path = most_recent_save_path.joinpath("params").joinpath(
            "best_params.h5"
        )

    if not specified_path.is_file():
        raise ValueError(
            f"specified path is neither an h5 path, nor a directory containing directories of h5: {specified_path}"
        )

    model.load_weights(str(specified_path))
    logger.info(f"params loaded from {specified_path}")

    # Right now, we're only going to add the first loss to the existing train
    # loop
    objective_list = list(perf_cdict["objectives"].keys())
    if len(objective_list) > 1:
        raise ValueError(
            "Currently, only one objective is supported by the training loop logic. There are {len(objective_list)} specified ({objective_list})"
        )
    first_and_only_obj = objective_list[0]

    # loss
    # get loss function
    loss_object = configure_loss(perf_cdict["objectives"][first_and_only_obj]["loss"])

    # mean loss
    avg_eval_loss = tf.keras.metrics.Mean(name="validation_loss", dtype=tf.float32)

    # metrics
    eval_metric_fns, metric_order = [], []
    # TODO: this is hardcoded to only the first objective
    met_opts = perf_cdict["objectives"][first_and_only_obj]["metric"]["options"]
    # TODO: this is hardcoded to only the first objective
    for i, metric in enumerate(
        perf_cdict["objectives"][first_and_only_obj]["metric"]["type"]
    ):
        try:
            met_opt_dict = met_opts[i]
        except TypeError:
            # no options
            met_opt_dict = None
        except IndexError:
            # No options for particular metric
            met_opt_dict = None
        eval_metric_fn = configure_metric(metric, met_opt_dict)
        eval_metric_fns.append(eval_metric_fn)
        metric_order.append(metric)

    # reset metrics (should already be reset)
    for eval_metric_fn in eval_metric_fns:
        eval_metric_fn.reset_states()

    # get datasets
    # TODO: ensure eval dataset is the same as used previously
    # TODO: apply shuffle/aug/reshape from config
    if not dataset:
        eval_ds = get_configured_dataset(data_cdict, hp_cdict)
    else:
        assert isinstance(
            dataset, tf.data.Dataset
        ), f"a {type(dataset)} was passed as a test dataset, please pass an instance of {tf.data.Dataset}"
        eval_ds = get_configured_dataset(data_cdict, hp_cdict, ds=dataset)

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
