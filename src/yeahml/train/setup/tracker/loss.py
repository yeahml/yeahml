from yeahml.train.setup.tracker.tracker import Tracker


def create_loss_trackers(
    optimizer_to_loss_name_map, ds_names=None, descriptions=None, to_track=None
):
    """creates a dictionary mapping the each loss by name to a Tracker for the 
    number of instances that have passed through the model during training
    
    Returns
    -------
    loss_dict_tracker
        holds loss_name: {Tracker()} for each loss for train and val
    """

    # TODO: The joint losses (if they exist) should also be tracked here

    if not isinstance(ds_names, list):
        if isinstance(ds_names, str):
            ds_names = [ds_names]
        else:
            raise TypeError(
                f"ds_names ({ds_names}) must be type string or list of string not {type(ds_names)}"
            )

    if not isinstance(descriptions, list):
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        else:
            raise TypeError(
                f"descriptions ({descriptions}) must be type string or list of string not {type(descriptions)}"
            )

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

    loss_dict_tracker = {}
    for _, temp_dict in optimizer_to_loss_name_map.items():
        for name in temp_dict["losses_to_optimize"]["names"]:
            loss_dict_tracker[name] = {}
            for ds_name in ds_names:
                loss_dict_tracker[name][ds_name] = {}
                for description in descriptions:
                    # 'mean', for example
                    loss_dict_tracker[name][ds_name][description] = Tracker(
                        to_track=to_track
                    )

    return loss_dict_tracker


def update_loss_trackers(
    ds_name, step_value, loss_tracker, loss_objective_names, loss_obj_descs
):
    """[summary]
    
    Parameters
    ----------
    ds_name : str
        the name of the dataset, e.g. "train"
    step_value : int
        the number of 'steps' which are currently the number of training
        instances that the model has been trained on # TODO: not optimizer specific
    loss_tracker : Dict[str(objective_name): Dict[str(ds_name): Dict[str(description): Tracker]]]
        dictionary containing all of the relevant Trackers
        example:
            # {
            #     "main_obj": {"train": {"mean": Tracker}, "val": {"mean": Tracker}},
            #     "second_obj": {"train": {"mean": Tracker}, "val": {"mean": Tracker}},
            # }
    loss_objective_names : List[str]
        list of the names of the objectives that need to be updated
        example: 
            ['main_obj']
    loss_obj_descs : Dict[str: List[tf_object]]
        dictionary of the description of objects to track and the corresponding tf_object
        example:
            # {"mean": [tf.mean()]} one item in teh list for each obj name
    
    Returns
    -------
    [type]
        [description]
    """

    update_dict = {}
    for i, name in enumerate(loss_objective_names):
        update_dict[name] = {}
        # loop objectives
        objective_trackers = loss_tracker[name][ds_name]
        for loss_obj_desc, tf_loss_objs in loss_obj_descs.items():
            # e.g. "mean", [tf.mean()] // where each tf corresponds to obj name
            cur_val = tf_loss_objs[i].result().numpy()
            cur_tracker = objective_trackers[loss_obj_desc]
            cur_update = cur_tracker.update(step=step_value, value=cur_val)
            update_dict[name][loss_obj_desc] = cur_update

    return update_dict

    # for name, mets in l2o_loss_record.items():
    #     # name is the name of {description?} of the metric (mean, etc.)
    #     if not isinstance(mets, list):
    #         mets = [mets]
    #     for i, metric_object in enumerate(mets):

    #         if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"]:
    #             loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"] = [
    #                 step_value
    #             ]
    #         else:
    #             loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
    #                 "steps"
    #             ].append(step_value)

    #         cur_val = metric_object.result().numpy()
    #         if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"]:
    #             loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"] = [
    #                 cur_val
    #             ]
    #         else:
    #             loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
    #                 "values"
    #             ].append(cur_val)

    #         prev_best = loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
    #             "best"
    #         ]
    #         if not prev_best:
    #             loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
    #                 "best"
    #             ] = cur_val
    #             update = True
    #         else:
    #             # NOTE: currently assuming min
    #             if cur_val < prev_best:
    #                 loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
    #                     "best"
    #                 ] = cur_val
    #                 update = True
    #             else:
    #                 update = False
    #         best_update[l2o_names[i]] = {name: update}
    #         # TODO: logic with the current best

    # return best_update


def record_losses(
    ds_name, step_name, step_value, loss_tracker, l2o_names, l2o_loss_record
):
    # TODO: I think we should change this logic such that we only keep track of
    # the tensor/metric name and use that as a lookup followed by a dict of
    # additional information?
    best_update = {}

    for name, mets in l2o_loss_record.items():
        # name is the name of {description?} of the metric (mean, etc.)
        if not isinstance(mets, list):
            mets = [mets]
        for i, metric_object in enumerate(mets):

            if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"]:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["steps"] = [
                    step_value
                ]
            else:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "steps"
                ].append(step_value)

            cur_val = metric_object.result().numpy()
            if not loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"]:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name]["values"] = [
                    cur_val
                ]
            else:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "values"
                ].append(cur_val)

            prev_best = loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                "best"
            ]
            if not prev_best:
                loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                    "best"
                ] = cur_val
                update = True
            else:
                # NOTE: currently assuming min
                if cur_val < prev_best:
                    loss_dict_tracker[l2o_names[i]][ds_name][name][step_name][
                        "best"
                    ] = cur_val
                    update = True
                else:
                    update = False
            best_update[l2o_names[i]] = {name: update}
            # TODO: logic with the current best

    return best_update
