from typing import Any, Dict, List

import tensorflow as tf

from yeahml.train.setup.tracker.tracker import Tracker


def create_loss_trackers(objective_dict):
    """creates a dictionary mapping the each loss by name to a Tracker for the 
    number of instances that have passed through the model during training

    NOTE: is there a better way to build up this dictionary?

    e.g.
        {
            "in_config": {
                "type": "supervised",
                "options": {"prediction": "dense_out", "target": "target_v"},
                "dataset": "abalone",
            },
            "loss": {
                "object": "<function mean_squared_error at 0x7f9ba45c78c8>",
                "track": {
                    "train": {
                        "mse": {
                            "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f9b805e8668>"
                        }},
                    "val": {
                        "mse": {
                            "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f9b805e8748>"
                        }},},
            },
            "metrics": {'...'},
        }
    
    Returns
    -------
    loss_dict_tracker
        e.g.
            {
                "main_obj": {
                    "abalone": {
                        "train": {
                            "mse": {
                                "mean": "<yeahml.train.setup.tracker.tracker.Tracker object at 0x7f9b80579588>"
                            }
                        },
                        "val": {
                            "mse": {
                                "mean": "<yeahml.train.setup.tracker.tracker.Tracker object at 0x7f9b8c0c27b8>"
            }},}}}

    """

    # get data set name, create empty tracker
    ds_name = objective_dict["in_config"]["dataset"]
    # loss_dict_tracker = {objective_name: {ds_name: {}}}
    loss_dict_tracker = {ds_name: {}}

    for ds_split_name, dd in objective_dict["loss"]["track"].items():
        loss_dict_tracker[ds_name][ds_split_name] = {}
        for loss_name, loss_track_conf in dd.items():
            loss_dict_tracker[ds_name][ds_split_name][loss_name] = {}
            assert (
                len(loss_track_conf) == 1
            ), f"{loss_track_conf} should only have one item"
            track_name, _ = next(iter(loss_track_conf.items()))
            # _ is the object

            # NOTE: assume that we are always minimizing the loss, thus
            # interested in tracking the minimum
            loss_dict_tracker[ds_name][ds_split_name][loss_name][track_name] = Tracker(
                to_track=["min"]
            )

    return loss_dict_tracker


def create_joint_loss_tracker(
    opt_name: str, objective_names: List[str], objectives_dict: Dict[str, Any]
):
    """[summary]
    
    Parameters
    ----------

    opt_name: str
        used for naming the joint metric
        e.g.
            "main_opt"
    loss_names : List[str]
        the names of objectives that contain losses that are optimized jointly
        e.g.
            ['main_obj', 'second_obj']
    objectives_dict : Dict[str, Any]
        [description]
        e.g.
            {
                "main_obj": {
                    "in_config": {
                        "type": "supervised",
                        "options": {"prediction": "dense_out", "target": "target_v"},
                        "dataset": "abalone",
                    },
                    "loss": {
                        "object": "tf.obj",
                        "track": {
                            "train": {"mse": {"mean": "tf.obj"}},
                            "val": {"mse": {"mean": "tf.obj"}},
                        },
                    },
                    "metrics": {"meansquarederror": {"train": "tf.obj", "val": "tf.obj"}},
                },
                "second_obj": {
                    "in_config": {
                        "type": "supervised",
                        "options": {"prediction": "dense_out", "target": "target_v"},
                        "dataset": "abalone",
                    },
                    "loss": {
                        "object": "tf.obj",
                        "track": {
                            "train": {"mae": {"mean": "tf.obj"}},
                            "val": {"mae": {"mean": "tf.obj"}},
                        },
                    },
                    "metrics": {"meanabsoluteerror": {"train": "tf.obj", "val": "tf.obj"}},
                },
            }
    
    Returns
    -------
    [type]
        [description]
        e.g.
            {
                "abalone": {
                    "val": {
                        "object": "<tensorflow.python.keras.metrics.Mean object at 0x7f9b804564a8>",
                        "track": "<yeahml.train.setup.tracker.tracker.Tracker object at 0x7f9b804569e8>",
            }}}
    
    """
    # The issue with the joint tracker is that there is no object present.. so
    # either we should
    # 1 - make one here and include it
    # 2 - make the metric/loss include the objects as well...
    # 3 - create the object elsewhere..
    # For now, I'm going to opt for 1...
    joint_dict_tracker = {}

    # TODO: in the future, we may need to check for ds name and split overlap
    # but currently we're only supporting the same ds name and splits
    # get matching dataset names and data split names to create joint trackers
    # for the overlap
    dd = {}
    for objective_name in objective_names:
        obj_dict = objectives_dict[objective_name]
        dd[objective_name] = {
            "ds_name": obj_dict["in_config"]["dataset"],
            "splits": list(obj_dict["loss"]["track"].keys()),
        }

    ds_name = None
    common_splits = None
    for k, vd in dd.items():
        name = vd["ds_name"]
        splits = vd["splits"]
        if not ds_name:
            ds_name = name
        else:
            if name != ds_name:
                raise (
                    f"the objectives {objective_names} correspond to different datasets."
                )
        if not common_splits:
            common_splits = splits
        else:
            for split in splits:
                if split not in common_splits:
                    raise (
                        f"the objectives {objective_names} correspond to different dataset splits."
                    )

    for split in common_splits:
        joint_dict_tracker = {
            ds_name: {
                split: {
                    "object": tf.keras.metrics.Mean(
                        name=f"{opt_name}__joint__{ds_name}__{split}", dtype=tf.float32
                    ),
                    "track": Tracker(to_track=["min"]),
                }
            }
        }

    return joint_dict_tracker
