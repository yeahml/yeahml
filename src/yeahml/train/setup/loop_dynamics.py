# from yeahml.build.components.optimizer import get_optimizer

from typing import Any, Dict

from yeahml.build.components.optimizer import return_optimizer
from yeahml.config.model.util import make_hash
from yeahml.train.setup.tracker.loss import (
    create_loss_trackers,
    create_joint_loss_tracker,
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


# def obtain_optimizer_loss_mapping(optimizers_dict, objectives_dict, dataset_names):

#     # print(optimizers_dict)
#     # print("******" * 8)
#     # print(objectives_dict)
#     # print("******" * 8)

#     # optimizer_to_loss_name_map = {}
#     # for cur_optimizer_name, optimizer_dict in optimizers_dict.items():
#     #     try:
#     #         objectives_to_opt = optimizer_dict["objectives"]
#     #     except KeyError:
#     #         raise KeyError(f"no objectives found for {cur_optimizer_name}")

#     #     for o in objectives_to_opt:
#     #         pass

#     # sys.exit()

#     # TODO: this function needs to be rewritten -- no hardcoding and better organization
#     # NOTE: multiple losses by the same optimizer, are currently only modeled
#     # jointly, if we wish to model the losses seperately (sequentially or
#     # alternating), then we would want to use a second optimizer
#     objectives_used = set()
#     optimizer_to_loss_name_map = {}
#     for cur_optimizer_name, optimizer_dict in optimizers_dict.items():
#         loss_names_to_optimize = []
#         loss_objs_to_optimize = []
#         train_means = []
#         val_means = []

#         try:
#             objectives_to_opt = optimizer_dict["objectives"]
#         except KeyError:
#             raise KeyError(f"no objectives found for {cur_optimizer_name}")

#         in_to_optimizer = None
#         for o in objectives_to_opt:
#             # add to set of all objectives used - for tracking purposes
#             objectives_used.add(o)

#             # sanity check ensure loss object from targeted objective exists
#             try:
#                 loss_object = objectives_dict[o]["loss"]["object"]
#             except KeyError:
#                 raise KeyError(f"no loss object is present in objective {o}")

#             try:
#                 train_mean = objectives_dict[o]["loss"]["train_mean"]
#             except KeyError:
#                 raise KeyError(f"no train_mean is present in objective {o}")

#             try:
#                 val_mean = objectives_dict[o]["loss"]["val_mean"]
#             except KeyError:
#                 raise KeyError(f"no val_mean is present in objective {o}")

#             try:
#                 in_conf = objectives_dict[o]["in_config"]
#             except NotImplementedError:
#                 raise NotImplementedError(
#                     f"no options present in {objectives_dict[o]['in_config']}"
#                 )

#             if in_to_optimizer:
#                 if not in_to_optimizer == in_conf:
#                     raise ValueError(
#                         f"The in to optimizer is {in_to_optimizer} but the in_conf for {o} is {in_conf}, they should be the same"
#                     )
#             else:
#                 in_to_optimizer = in_conf

#             # add loss object to a list for grouping
#             loss_names_to_optimize.append(o)
#             loss_objs_to_optimize.append(loss_object)
#             train_means.append(train_mean)
#             val_means.append(val_mean)

#         # TODO: this may not be true -- there may only be one loss to optimize
#         # create and include joint metric
#         joint_name = "__".join(loss_names_to_optimize) + "__joint"
#         train_name = joint_name + "_train"
#         val_name = joint_name + "_val"
#         joint_object_train = tf.keras.metrics.Mean(name=train_name, dtype=tf.float32)
#         joint_object_val = tf.keras.metrics.Mean(name=val_name, dtype=tf.float32)

#         optimizer_to_loss_name_map[cur_optimizer_name] = {
#             "losses_to_optimize": {
#                 "names": loss_names_to_optimize,
#                 "objects": loss_objs_to_optimize,
#                 "record": {"train": {"mean": train_means}, "val": {"mean": val_means}},
#                 "joint_name": train_name,
#                 "joint_record": {
#                     "train": {"mean": joint_object_train},
#                     "val": {"mean": joint_object_val},
#                 },
#             },
#             "in_conf": in_conf,
#         }

#     # ensure all losses are mapped to an optimizer
#     obj_not_used = []
#     for objective_name, obj_dict in objectives_dict.items():
#         # only add objective if it contains a loss
#         try:
#             _ = obj_dict["loss"]
#             if objective_name not in objectives_used:
#                 obj_not_used.append(objective_name)
#         except KeyError:
#             pass
#     if obj_not_used:
#         raise ValueError(f"objectives {obj_not_used} are not mapped to an optimizer")

#     return optimizer_to_loss_name_map


# def map_in_config_to_objective(objectives_dict):
#     in_hash_to_objectives = {}
#     for o, d in objectives_dict.items():
#         in_conf = d["in_config"]
#         in_conf_hash = make_hash(in_conf)
#         try:
#             stored_conf = in_hash_to_objectives[in_conf_hash]["in_config"]
#             if not stored_conf == in_conf:
#                 raise ValueError(
#                     f"the hash is the same, but the in config is different..."
#                 )
#         except KeyError:
#             in_hash_to_objectives[in_conf_hash] = {"in_config": in_conf}

#         # ? is there a case where there is no objective?
#         try:
#             stored_objectives = in_hash_to_objectives[in_conf_hash]["objectives"]
#             stored_objectives.append(o)
#         except KeyError:
#             in_hash_to_objectives[in_conf_hash]["objectives"] = [o]

#     return in_hash_to_objectives


# def create_grouped_metrics(
#     objectives_dict: Dict[str, Any], in_hash_to_objectives: Dict[int, Dict[str, Any]]
# ) -> Dict[int, Any]:
#     """[summary]

#     Parameters
#     ----------
#     objectives_dict : Dict[str, Any]
#         [description]
#         e.g.
#             {
#                 "main_obj": {
#                     "in_config": {
#                         "type": "supervised",
#                         "options": {"prediction": "dense_out", "target": "target_v"},
#                         "dataset": "abalone",
#                     },
#                     "loss": {
#                         "object": "<function mean_squared_error at 0x7fa5ab24b268>",
#                         "track": {
#                             "train": {
#                                 "mse": {
#                                     "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7fa580348f60>"
#                                 },
#                             "val": {
#                                 "mse": {
#                                     "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7fa580348ef0>"
#                                 }},},},
#                     "metrics": {
#                         "meansquarederror": {
#                             "train": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7fa5802dd630>",
#                             "val": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7fa5802dda90>",
#                 }},
#                 ...
#                 }}}
#     in_hash_to_objectives : Dict[int, Dict[str, Any]]
#         [description]
#         e.g.
#             {
#                 7236791920024407028: {
#                     "in_config": {
#                         "type": "supervised",
#                         "options": {"prediction": "dense_out", "target": "target_v"},
#                     },
#                     "objectives": ["main_obj", "second_obj"],
#             }}

#     Returns
#     -------
#     Dict[int, Any]
#         [description]
#     """
#     print("######" * 8)
#     print(objectives_dict)
#     print("*****" * 8)
#     print(in_hash_to_objectives)
#     print("######" * 8)

#     in_hash_to_metrics_config = {}

#     # loop the different in/out combinations and build metrics for each
#     # this dict may become a bit messy because there is the train+val to keep
#     # track of
#     for k, v in in_hash_to_objectives.items():
#         in_hash_to_metrics_config[k] = {"in_config": v["in_config"]}
#         in_hash_to_metrics_config[k]["metric_order"] = []
#         in_hash_to_metrics_config[k]["objects"] = {"train": [], "val": []}
#         for objective in v["objectives"]:
#             obj_dict = objectives_dict[objective]
#             try:
#                 cur_metrics = obj_dict["metrics"]
#             except KeyError:
#                 cur_metrics = None

#             if cur_metrics:
#                 in_hash_to_metrics_config[k]["metric_order"].extend(
#                     obj_dict["metrics"]["metric_order"]
#                 )
#                 in_hash_to_metrics_config[k]["objects"]["train"].extend(
#                     obj_dict["metrics"]["train_metrics"]
#                 )
#                 in_hash_to_metrics_config[k]["objects"]["val"].extend(
#                     obj_dict["metrics"]["val_metrics"]
#                 )

#     print(in_hash_to_objectives)
#     print("xxxxxxxx" * 8)

#     return in_hash_to_metrics_config


def _return_loss_trackers(raw_obj_dict):
    try:
        raw_loss_conf = raw_obj_dict["loss"]
    except KeyError:
        raw_loss_conf = None

    if raw_loss_conf:
        loss_tracker_dict = create_loss_trackers(raw_loss_conf, raw_obj_dict)
    else:
        loss_tracker_dict = None

    return loss_tracker_dict


def _return_metric_trackers(raw_obj_dict):
    try:
        raw_metric_conf = raw_obj_dict["metric"]
    except KeyError:
        raw_metric_conf = None

    if raw_metric_conf:
        metric_tracker_dict = create_metric_trackers(raw_metric_conf, raw_obj_dict)
    else:
        metric_tracker_dict = None

    return metric_tracker_dict


def create_full_dict(optimizers_dict=None, objectives_dict=None, datasets_dict=None):
    # optimizers_dict
    # {optimizer_name: {"optimizer": tf.obj, "objective": [objective_name]}}

    # objective_dict
    # {objective_name: "in_config": {...}, "loss": {...}, "metric": {...}}

    # dataset_dict
    # {'abalone': {'train': 'tf.Data.dataset', 'val': 'tf.Data.dataset'}}
    ret_dict = {}
    for opt_name, opt_and_obj_dict in optimizers_dict.items():
        ret_dict[opt_name] = {**opt_and_obj_dict}
        objectives = opt_and_obj_dict["objectives"]
        loss_names = []
        for objective_name in objectives:
            try:
                raw_obj_dict = objectives_dict[objective_name]
            except KeyError:
                raise KeyError(
                    f"objective name {objective_name} not found in objective_dict :{objectives_dict.keys()}"
                )

            loss_tracker_dict = _return_loss_trackers(raw_obj_dict)
            if loss_tracker_dict:
                loss_names.append(objective_name)
            metric_tracker_dict = _return_metric_trackers(raw_obj_dict)
        if len(loss_names) > 1:
            # there are more than 1 losses present, create joint tracker for
            # both losses
            joint_tracker_dict = create_joint_loss_tracker(loss_names, raw_obj_dict)

        ret_dict[opt_name] = {
            "loss": loss_tracker_dict,
            "metric": metric_tracker_dict,
            "joint": joint_tracker_dict,
        }

    print(ret_dict)
    return ret_dict


# obj_dict = {
#     "in_config": {
#         "type": "supervised",
#         "options": {"prediction": "dense_out", "target": "target_v"},
#         "dataset": "abalone",
#     },
#     "loss": {
#         "object": "<function mean_squared_error at 0x7f73c8cf98c8>",
#         "track": {
#             "train": {
#                 "mse": {
#                     "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f73a0448e80>"
#                 }
#             },
#             "val": {
#                 "mse": {
#                     "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f73a0448f60>"
#                 }
#             },
#         },
#     },
#     "metrics": {
#         "meansquarederror": {
#             "train": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f73a0448f98>",
#             "val": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f73a0460898>",
#         }
#     },
# }
