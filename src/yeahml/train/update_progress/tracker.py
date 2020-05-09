def update_metrics_tracking(
    metrics_objective_names,
    objectives_dict,
    opt_tracker_dict,
    obj_to_grads,
    num_train_instances,
    num_training_ops,
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
                    value=result, step=num_train_instances, global_step=num_training_ops
                )
                update_dict[cur_objective][metric_name] = cur_update

    return update_dict


def update_val_loss_trackers(
    cur_loss_conf, cur_loss_tracker_dict, num_train_instances, num_training_ops
):
    # update Tracker and reset tf states

    update_dict = {}
    for loss_name, desc_dict in cur_loss_conf.items():
        update_dict[loss_name] = {}
        for desc_name, desc_tf_obj in desc_dict.items():
            desc_tracker = cur_loss_tracker_dict[loss_name][desc_name]

            tf_desc_val = desc_tf_obj.result().numpy()
            desc_tf_obj.reset_states()

            cur_update = desc_tracker.update(
                value=tf_desc_val,
                step=num_train_instances,
                global_step=num_training_ops,
            )
            update_dict[loss_name][desc_name] = cur_update

    return update_dict


def update_val_metrics_trackers(
    metrics_conf,
    cur_metric_tracker_dict,
    val_name,
    num_train_instances,
    num_training_ops,
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
            cur_update = metric_tracker.update(
                value=result, step=num_train_instances, global_step=num_training_ops
            )
            update_dict[metric_name] = cur_update

    return update_dict
