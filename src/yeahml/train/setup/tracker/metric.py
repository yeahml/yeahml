from yeahml.train.setup.tracker.tracker import Tracker


def create_metric_trackers_v1(in_hash_to_metrics_config, to_track=None):
    """
    # TODO: the to_track should be metric/loss specific

    """

    # TODO: this should match the losses, not have a separate process
    # ds_names=None,
    # if not isinstance(ds_names, list):
    #     if isinstance(ds_names, str):
    #         ds_names = [ds_names]
    #     else:
    #         raise TypeError(
    #             f"ds_names ({ds_names}) must be type string or list of string not {type(ds_names)}"
    #         )

    if not to_track:
        to_track = ["max", "min"]
    if not isinstance(to_track, list):
        if isinstance(to_track, str):
            to_track = [to_track]
        else:
            raise TypeError(
                f"to_track ({to_track}) must be type string or list of string not {type(to_track)}"
            )
    ALLOWED_TO_TRACK = ["max", "min"]
    to_track = [name.lower() for name in to_track]
    for name in to_track:
        if name not in ALLOWED_TO_TRACK:
            raise ValueError(
                f"{name} is not allowed to be track. please only use from selected: {ALLOWED_TO_TRACK}"
            )

    metric_dict_tracker = {}
    for _, temp_dict in in_hash_to_metrics_config.items():

        try:
            metric_names = temp_dict["metric_order"]
            metric_dict = temp_dict["objects"]
        except KeyError:
            metric_dict = None
            pass

        if metric_dict:
            for ds_name, metric_list in metric_dict.items():
                # TODO: why isn't there an "objective name?"
                metric_dict_tracker[ds_name] = {}
                for i, metric in enumerate(metric_list):
                    metric_dict_tracker[ds_name][metric_names[i]] = Tracker(
                        to_track=to_track
                    )

    return metric_dict_tracker


def create_metric_trackers(in_hash_to_metrics_config):
    """
    # TODO: the to_track should be metric/loss specific

    """

    # TODO: this should match the losses, not have a separate process
    # ds_names=None,
    # if not isinstance(ds_names, list):
    #     if isinstance(ds_names, str):
    #         ds_names = [ds_names]
    #     else:
    #         raise TypeError(
    #             f"ds_names ({ds_names}) must be type string or list of string not {type(ds_names)}"
    #         )

    # if not to_track:
    #     to_track = ["max", "min"]
    # if not isinstance(to_track, list):
    #     if isinstance(to_track, str):
    #         to_track = [to_track]
    #     else:
    #         raise TypeError(
    #             f"to_track ({to_track}) must be type string or list of string not {type(to_track)}"
    #         )
    # ALLOWED_TO_TRACK = ["max", "min"]
    # to_track = [name.lower() for name in to_track]
    # for name in to_track:
    #     if name not in ALLOWED_TO_TRACK:
    #         raise ValueError(
    #             f"{name} is not allowed to be track. please only use from selected: {ALLOWED_TO_TRACK}"
    #         )

    # metric_dict_tracker = {}
    # for _, temp_dict in in_hash_to_metrics_config.items():

    #     try:
    #         metric_names = temp_dict["metric_order"]
    #         metric_dict = temp_dict["objects"]
    #     except KeyError:
    #         metric_dict = None
    #         pass

    #     if metric_dict:
    #         for ds_name, metric_list in metric_dict.items():
    #             # TODO: why isn't there an "objective name?"
    #             metric_dict_tracker[ds_name] = {}
    #             for i, metric in enumerate(metric_list):
    #                 metric_dict_tracker[ds_name][metric_names[i]] = Tracker(
    #                     to_track=to_track
    #                 )

    raise NotImplementedError(f"not implemented yet...")
    # return metric_dict_tracker
    return None


def update_metric_trackers(
    ds_name, step_value, metric_trackers, metric_names, tf_metric_objects
):
    # {
    #     "train": {
    #         "meansquarederror": Tracker,
    #         "meanabsoluteerror": Tracker,
    #     },
    #     "val": {
    #         "meansquarederror": Tracker,
    #         "meanabsoluteerror": Tracker,
    #     },
    # }

    update_dict = {}
    ds_trackers = metric_trackers[ds_name]
    # loop objectives
    for i, name in enumerate(metric_names):
        cur_val = tf_metric_objects[i].result().numpy()
        cur_tracker = ds_trackers[name]
        cur_update = cur_tracker.update(step=step_value, value=cur_val)
        update_dict[name] = cur_update

    return update_dict
