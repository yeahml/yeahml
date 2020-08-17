from yeahml.train.update_progress.tf_objectives import (
    update_tf_val_metrics,
    update_supervised_tf_metrics,
)
from yeahml.train.update_progress.tracker import (
    update_loss_trackers,
    update_val_metrics_trackers,
)
from yeahml.train.util import get_losses_to_update, get_next_batch, re_init_iter
import tensorflow as tf


def inference_dataset(
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
    num_training_ops,
    objective_to_output_index,
    objectives_dict,
    v_writer,
    logger,
    split_name,
):
    logger.debug(
        f"START inference on {split_name}, num_train_instances: {num_train_instances}"
    )

    cur_update = {}
    for cur_objective in loss_objective_names:
        cur_obj_output_index = objective_to_output_index[cur_objective]
        cur_in_conf = objectives_dict[cur_objective]["in_config"]
        cur_ds_name = cur_in_conf["dataset"]
        cur_ds_iter_dict = dataset_iter_dict[cur_ds_name]
        if split_name not in cur_ds_iter_dict.keys():
            raise ValueError(
                f"{cur_in_conf['dataset']} does not have a '{split_name}' dataset"
            )
        cur_val_iter = cur_ds_iter_dict[split_name]

        # loss
        loss_conf = objectives_dict[cur_objective]["loss"]

        tf_val_loss_descs_to_update = get_losses_to_update(loss_conf, split_name)

        # metrics
        metrics_conf = objectives_dict[cur_objective]["metrics"]

        # iterate batches
        cur_batch = get_next_batch(cur_val_iter)

        while cur_batch:

            val_dict = cur_val_fn(
                model,
                cur_batch,
                loss_conf["object"],
                cur_obj_output_index,
                tf_val_loss_descs_to_update,
            )
            # {"predictions": prediction,
            # "final_loss": final_loss,
            # "losses": loss,
            # "y_batch": y_batch,}

            # update tf objects
            # update_tf_loss_descriptions(val_dict, tf_val_loss_descs_to_update)
            update_tf_val_metrics(
                val_dict, metrics_conf, split_name, cur_in_conf["type"]
            )

            # next batch until end
            cur_batch = get_next_batch(cur_val_iter)

        # reinitialize validation iterator
        dataset_iter_dict[cur_ds_name][split_name] = re_init_iter(
            cur_ds_name, split_name, dataset_dict
        )

        # update trackers
        cur_loss_tracker_dict = opt_tracker_dict[cur_objective]["loss"][cur_ds_name][
            split_name
        ]
        cur_loss_update = update_loss_trackers(
            loss_conf["track"][split_name],
            cur_loss_tracker_dict,
            num_train_instances,
            num_training_ops,
            tb_writer=v_writer,
            ds_name=cur_ds_name,
            objective_name=cur_objective,
        )

        # metrics are optional -- there many only be a loss
        if metrics_conf:
            cur_metric_tracker_dict = opt_tracker_dict[cur_objective]["metrics"][
                cur_ds_name
            ][split_name]
            cur_metrics_update = update_val_metrics_trackers(
                metrics_conf,
                cur_metric_tracker_dict,
                split_name,
                num_train_instances,
                num_training_ops,
                tb_writer=v_writer,
                ds_name=cur_ds_name,
                objective_name=cur_objective,
            )
        else:
            cur_metrics_update = None

        cur_update[cur_objective] = {
            "loss": cur_loss_update,
            "metrics": cur_metrics_update,
        }

    logger.info(f"done inference on {split_name} - {num_train_instances}")
    return cur_update


def inference_on_ds(
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
    pred_dict=None,
):
    split_name = eval_split
    logger.debug(f"START inference_on_ds on {split_name}")

    # losses
    # TODO: this essentially only allows one description per loss.. which seems
    # than ideal.. I think the soution could be 1: naming or 2:
    # losses:[a,b], descs:[[a_desc_1,a_desc_2], [b_desc]] then inner loop
    loss_objects, loss_descriptions = [], []
    for ln in cur_loss_objective_names:
        loss_conf = objectives_to_objects[ln]["loss"]
        loss_obj = loss_conf["object"]
        tf_val_loss_descs_to_update = get_losses_to_update(loss_conf, split_name)
        loss_objects.append(loss_obj)
        loss_descriptions.append(tf_val_loss_descs_to_update)

    supervised_met_objects = []
    for mn in cur_metrics_objective_names:
        assert (
            objectives_to_objects[mn]["in_config"]["type"] == "supervised"
        ), f"only supervised is currently supported :( not {objectives_to_objects[mn]['in_config']['type']}"
        metric_conf = objectives_to_objects[mn]["metrics"]
        for _, split_to_metric in metric_conf.items():
            # _ is `metric_name`
            if split_name in split_to_metric.keys():
                metric_tf_obj = split_to_metric[split_name]
                supervised_met_objects.append(metric_tf_obj)

    cur_batch = get_next_batch(cur_dataset_iter)

    temp_ret = None
    if pred_dict:
        temp_ret = {"pred": [], "target": []}
        try:
            pred_fn = pred_dict["pred"]["fn"]
            pred_options = pred_dict["pred"]["options"]
        except KeyError:
            pred_fn, pred_options = None, None
        logger.debug(f"pred_fn set to {pred_fn}, options: {pred_options}")

        try:
            target_fn = pred_dict["target"]["fn"]
            target_options = pred_dict["target"]["options"]
        except KeyError:
            target_fn, target_options = None, None
        logger.debug(f"target_fn set to {target_fn}, options: {target_options}")

    while cur_batch:

        inference_dict = cur_inference_fn(
            model, cur_batch, loss_objects, cur_pred_index, loss_descriptions
        )
        # {"predictions": prediction,
        # "final_loss": final_loss,
        # "losses": loss,
        # "y_batch": y_batch,}

        if pred_dict:
            pred_raw = inference_dict["predictions"]
            if pred_fn:
                pred_raw = pred_fn(pred_raw, **pred_options)
            preds_ = pred_raw.numpy().flatten()

            target_raw = inference_dict["y_batch"]
            if target_fn:
                target_raw = target_fn(target_raw, **target_options)
            targets_ = target_raw.numpy().flatten()

            assert len(preds_) == len(
                targets_
            ), f"targets and predictions are of different sizes pred:{len(preds_)} target:{len(targets_)}"

            temp_ret["pred"].extend(preds_)
            temp_ret["target"].extend(targets_)

        # update tf objects
        # update_tf_loss_descriptions(val_dict, tf_val_loss_descs_to_update)
        update_supervised_tf_metrics(inference_dict, supervised_met_objects)

        # next batch until end
        cur_batch = get_next_batch(cur_dataset_iter)

    out = {}
    out["loss"] = {}
    for i, loss_obj in enumerate(loss_objects):
        out["loss"][loss_obj.__name__] = {}
        loss_descs = loss_descriptions[i]
        for ld in loss_descs:
            out["loss"][loss_obj.__name__][ld.name] = ld.result().numpy()

    out["metrics"] = {}
    for met_obj in supervised_met_objects:
        out["metrics"][met_obj.name] = met_obj.result().numpy()

    if temp_ret:
        out["out"] = temp_ret

    logger.info("done iinference_on_ds")
    return out
