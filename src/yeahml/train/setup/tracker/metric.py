from typing import Any, Dict

from yeahml.train.setup.tracker.tracker import Tracker


def create_metric_trackers(objective_dict: Dict[str, Any]):
    """[summary]
    
    NOTE: is there a better way to build up this dictionary?

    Parameters
    ----------
    objective_name : str
        name of the objective
        e.g.
            "main_obj"
    objective_dict : Dict[str, Any]
        [description]
        e.g.
            {
                "in_config": {
                    "type": "supervised",
                    "options": {"prediction": "dense_out", "target": "target_v"},
                    "dataset": "abalone",
                },
                "loss": {"..."},
                "metrics": {
                    "meansquarederror": {
                        "train": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f9b8f6e63c8>",
                        "val": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f9b80314b70>",
                    }
                },
            }
    
    Returns
    -------
    [type]
        [description]
        e.g.
            {
                "main_obj": {
                    "abalone": {
                        "train": {
                            "meansquarederror": "<yeahml.train.setup.tracker.tracker.Tracker object at 0x7f9b80291940>"
                        },
                        "val": {
                            "meansquarederror": "<yeahml.train.setup.tracker.tracker.Tracker object at 0x7f9b80291a90>"
            },}}}
    
    """

    ds_name = objective_dict["in_config"]["dataset"]
    # metric_dict_tracker = {objective_name: {ds_name: {}}}
    metric_dict_tracker = {ds_name: {}}

    for metric_name, dd in objective_dict["metrics"].items():
        for ds_split_name, _ in dd.items():
            metric_dict_tracker[ds_name][ds_split_name] = {}
            # _ is the object

            # TODO: consider adding to_track to metrics?
            metric_dict_tracker[ds_name][ds_split_name][metric_name] = Tracker(
                to_track=["min", "max"]
            )

    return metric_dict_tracker


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
