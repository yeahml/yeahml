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
        if metric_conf:
            # TODO: this can be combined with the fn below
            for metric_name, split_to_metric in metric_conf.items():
                if split_name in split_to_metric.keys():
                    metric_obj = split_to_metric[split_name]

                    # TODO: hardcoded
                    if cur_in_conf["type"] == "supervised":
                        preds = obj_to_grads[cur_objective]["predictions"]
                        y_batch = obj_to_grads[cur_objective]["y_batch"]
                        metric_obj.update_state(y_batch, preds)


def update_tf_val_metrics(val_preds_dict, metrics_conf, val_name, cur_metrics_type):

    if metrics_conf:
        # TODO: this can be combined with the fn above
        for metric_name, split_to_metric in metrics_conf.items():
            if val_name in split_to_metric.keys():
                metric_tf_obj = split_to_metric[val_name]

                # TODO: hardcoded - some may not be a prediction/ground truth
                if cur_metrics_type == "supervised":
                    preds = val_preds_dict["predictions"]
                    y_batch = val_preds_dict["y_batch"]
                    metric_tf_obj.update_state(y_batch, preds)
