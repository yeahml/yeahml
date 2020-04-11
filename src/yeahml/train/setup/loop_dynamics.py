# from yeahml.build.components.optimizer import get_optimizer


from yeahml.build.components.optimizer import return_optimizer
from yeahml.train.setup.tracker.loss import (
    create_joint_loss_tracker,
    create_loss_trackers,
)
from yeahml.train.setup.tracker.metric import create_metric_trackers


def get_optimizers(optim_cdict):
    def _configure_optimizer(opt_dict):
        # TODO: this should not be here. (follow template for losses)
        optim_dict = return_optimizer(opt_dict["type"])
        optimizer = optim_dict["function"]

        # configure optimizers
        temp_dict = opt_dict.copy()
        optimizer = optimizer(**temp_dict["options"])

        return optimizer

    optimizers_dict = {}
    for opt_name, opt_dict in optim_cdict["optimizers"].items():
        configured_optimizer = _configure_optimizer(opt_dict)
        optimizers_dict[opt_name] = {
            "optimizer": configured_optimizer,
            "objectives": opt_dict["objectives"],
        }

    return optimizers_dict


def _return_loss_trackers(objective_name, raw_obj_dict):

    if "loss" in raw_obj_dict.keys():
        loss_tracker_dict = create_loss_trackers(objective_name, raw_obj_dict)
    else:
        loss_tracker_dict = None

    return loss_tracker_dict


def _return_metric_trackers(objective_name, raw_obj_dict):

    if "metrics" in raw_obj_dict.keys():
        metric_tracker_dict = create_metric_trackers(objective_name, raw_obj_dict)
    else:
        metric_tracker_dict = None

    return metric_tracker_dict


def create_full_dict(optimizers_dict=None, objectives_dict=None, datasets_dict=None):
    """[summary]
    
    Parameters
    ----------
    optimizers_dict : [type], optional
        [description], by default None
        e.g.
            {optimizer_name: {"optimizer": tf.obj, "objective": [objective_name]}}
    objectives_dict : [type], optional
        [description], by default None
        e.g.
            {objective_name: "in_config": {...}, "loss": {...}, "metric": {...}}
    datasets_dict : [type], optional
        [description], by default None
        e.g.
            {'abalone': {'train': 'tf.Data.dataset', 'val': 'tf.Data.dataset'}}
    
    Returns
    -------
    [type]
        [description]
        e.g.
            {
                "main_opt": {
                    "loss": {
                        "main_obj": {
                            "abalone": {
                                "train": {"mse": {"mean": "yml.Tracker"}},
                                "val": {"mse": {"mean": "yml.Tracker"}},
                            }
                        }
                    },
                    "metric": {
                        "main_obj": {
                            "abalone": {
                                "train": {"meansquarederror": "yml.Tracker"},
                                "val": {"meansquarederror": "yml.Tracker"},
                            }
                        }
                    },
                    "joint": None,
                },
                "second_opt": {
                    "loss": {
                        "second_obj": {
                            "abalone": {
                                "train": {"mae": {"mean": "yml.Tracker"}},
                                "val": {"mae": {"mean": "yml.Tracker"}},
                            }
                        }
                    },
                    "metric": {
                        "second_obj": {
                            "abalone": {
                                "train": {"meanabsoluteerror": "yml.Tracker"},
                                "val": {"meanabsoluteerror": "yml.Tracker"},
                            }
                        }
                    },
                    "joint": None,
                },
            }
    
    Raises
    ------
    KeyError
        [description]
    """

    ret_dict = {}
    for opt_name, opt_and_obj_dict in optimizers_dict.items():
        ret_dict[opt_name] = {**opt_and_obj_dict}
        objectives = opt_and_obj_dict["objectives"]

        # loss_names will be used to determine whether a joint tracker should be
        # created
        objective_names_with_losses = []
        for objective_name in objectives:
            try:
                raw_obj_dict = objectives_dict[objective_name]
            except KeyError:
                raise KeyError(
                    f"objective name {objective_name} not found in objective_dict :{objectives_dict.keys()}"
                )

            # create loss Trackers
            loss_tracker_dict = _return_loss_trackers(objective_name, raw_obj_dict)
            if loss_tracker_dict:
                objective_names_with_losses.append(objective_name)

            # create metric Trackers
            metric_tracker_dict = _return_metric_trackers(objective_name, raw_obj_dict)

        joint_tracker_dict = None
        if len(objective_names_with_losses) > 1:
            # there are more than 1 losses present, create joint tracker for
            # both losses
            joint_tracker_dict = create_joint_loss_tracker(
                opt_name, objective_names_with_losses, objectives_dict
            )

        ret_dict[opt_name] = {
            "loss": loss_tracker_dict,
            "metric": metric_tracker_dict,
            "joint": joint_tracker_dict,
        }

    return ret_dict
