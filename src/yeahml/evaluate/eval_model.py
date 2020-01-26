import os
import pathlib
from typing import Any, Dict

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

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]
    hp_cdict: Dict[str, Any] = config_dict["hyper_parameters"]
    perf_cdict: Dict[str, Any] = config_dict["performance"]

    logger = config_logger(model_cdict["model_root_dir"], log_cdict, "eval")

    # load best weights
    # TODO: load specific weights according to a param
    if weights_path:
        specified_path = weights_path
    else:
        # TODO: this needs to allow an easier specification for which params to load
        # The issue here is that if someone alters the model (perhaps in a notebook), then
        # retrains the model (but doesn't update the config), the "old" model, not new model
        # will be evaluated.  This will need to change
        if pathlib.Path(model_cdict["save/params"]).is_file():
            specified_path = model_cdict["save/params"]
        elif pathlib.Path(model_cdict["save/params"]).is_dir():
            p = pathlib.Path(model_cdict["save/params"])
            sub_dirs = [x for x in p.iterdir() if x.is_dir()]
            most_recent_subdir = sub_dirs[0]
            # TODO: this assumes .h5 is the filetype and that model.h5 is present
            specified_path = os.path.join(most_recent_subdir, "best_params.h5")
        else:
            raise ValueError(
                f"specified path is neither an h5 path, nor a directory containing directories of h5: {meta_cdict['save_weights_dir']}"
            )

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
