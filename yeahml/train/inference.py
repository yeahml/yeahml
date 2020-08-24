from yeahml.train.update_progress.tf_objectives import update_tf_val_metrics
from yeahml.train.update_progress.tracker import (
    update_loss_trackers,
    update_val_metrics_trackers,
)
from yeahml.train.util import get_losses_to_update, get_next_batch, re_init_iter


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
    logger.debug(f"START iterating validation - epoch: {num_train_instances}")

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

    logger.info(f"done validation - {num_train_instances}")
    return cur_update
