# from yeahml.build.components.optimizer import get_optimizer
import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metric import configure_metric
from yeahml.build.components.optimizer import return_optimizer
from yeahml.config.model.util import make_hash

# TODO: I'm not sure where this belongs yet
ACCEPTED_LOSS_TRACK = {"mean": tf.keras.metrics.Mean}


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


def _get_metrics(metric_config, datasets):

    assert len(metric_config["options"]) == len(
        metric_config["type"]
    ), f"len of options does not len of metrics: {len(metric_config['options'])} != {len(metric_config['type'])}"

    # loop operations and options
    metric_ds_to_metric = {}
    try:
        met_opts = metric_config["options"]
    except KeyError:
        met_opts = None
    for i, metric in enumerate(metric_config["type"]):
        metric_ds_to_metric[metric] = {}
        if met_opts:
            met_opt_dict = met_opts[i]
        else:
            met_opt_dict = None

        for dataset in datasets:
            metric_fn = configure_metric(metric, met_opt_dict)
            metric_ds_to_metric[metric][dataset] = metric_fn

    return metric_ds_to_metric


def get_objectives(objectives, datasets):
    # TODO: should these be grouped based on their inputs?

    if not isinstance(datasets, list):
        if not isinstance(datasets, str):
            raise ValueError(
                f"datasets ({datasets}) must be of type list of strings or string not {type(datasets)}"
            )
    else:
        for o in datasets:
            if not isinstance(o, str):
                raise ValueError(
                    f"object ({o}) in ({datasets}) must be of type string not {type(o)}"
                )

    obj_conf = {}
    for objective_name, objective_config in objectives.items():
        # in_config: defines the type (e.g. supervised) and io (e.g. pred, gt)
        # loss --> defines whether a loss is present
        # metric --> defines whether a metric is present
        in_config = objective_config["in_config"]

        try:
            loss_config = objective_config["loss"]
        except KeyError:
            loss_config = None

        try:
            metric_config = objective_config["metric"]
        except KeyError:
            metric_config = None

        if not loss_config and not metric_config:
            raise ValueError(
                f"Neither a loss or metric was defined for {objective_name}"
            )

        if loss_config:
            loss_ds_to_tracker = {}
            # build loss object
            loss_object = configure_loss(loss_config)
            loss_ds_to_tracker["object"] = loss_object

            try:
                loss_track = loss_config["track"]
                loss_ds_to_tracker["track"] = {}
            except KeyError:
                loss_track = None

            if loss_track:
                for name in loss_track:
                    try:
                        track_class = ACCEPTED_LOSS_TRACK[name]
                    except KeyError:
                        raise KeyError(
                            f"{name} is not an accepted track method please select from {ACCEPTED_LOSS_TRACK.keys()}"
                        )

                    for dataset in datasets:
                        loss_ds_to_tracker["track"][dataset] = {}
                        loss_ds_to_tracker["track"][dataset][loss_config["type"]] = {}
                        tf_tracker_name = f"loss_{objective_name}_{loss_config['type']}_{name}_{dataset}"
                        dtype = tf.float32
                        tf_tracker = track_class(name=tf_tracker_name, dtype=dtype)
                        loss_ds_to_tracker["track"][dataset][loss_config["type"]][
                            name
                        ] = tf_tracker

                    # avg_train_loss = tf.keras.metrics.Mean(
                    #     name=f"loss_mean_{objective_name}_{loss_type}_train",
                    #     dtype=tf.float32,
                    # )
                    # avg_val_loss = tf.keras.metrics.Mean(
                    #     name=f"loss_mean_{objective_name}_{loss_type}_validation",
                    #     dtype=tf.float32,
                    # )
        else:
            loss_ds_to_tracker = None

        if metric_config:
            metric_ds_to_metric = _get_metrics(metric_config, datasets)
        else:
            metric_ds_to_metric = None

        obj_conf[objective_name] = {
            "in_config": in_config,
            "loss": loss_ds_to_tracker,
            "metrics": metric_ds_to_metric,
        }

    # Currently, only supervised is accepted
    for objective_name, obj_dict in obj_conf.items():
        if obj_dict["in_config"]["type"] != "supervised":
            raise NotImplementedError(
                f"only 'supervised' is accepted as the type for the in_config of {objective_name}, not {obj_conf['in_config']['type']} yet..."
            )

    return obj_conf


def obtain_optimizer_loss_mapping(optimizers_dict, objectives_dict, datasets):

    print(optimizers_dict)
    print("******" * 8)
    print(objectives_dict)
    print("******" * 8)
    sys.exit()

    # TODO: this function needs to be rewritten -- no hardcoding and better organization
    # NOTE: multiple losses by the same optimizer, are currently only modeled
    # jointly, if we wish to model the losses seperately (sequentially or
    # alternating), then we would want to use a second optimizer
    objectives_used = set()
    optimizer_to_loss_name_map = {}
    for cur_optimizer_name, optimizer_dict in optimizers_dict.items():
        loss_names_to_optimize = []
        loss_objs_to_optimize = []
        train_means = []
        val_means = []

        try:
            objectives_to_opt = optimizer_dict["objectives"]
        except KeyError:
            raise KeyError(f"no objectives found for {cur_optimizer_name}")

        in_to_optimizer = None
        for o in objectives_to_opt:
            # add to set of all objectives used - for tracking purposes
            objectives_used.add(o)

            # sanity check ensure loss object from targeted objective exists
            try:
                loss_object = objectives_dict[o]["loss"]["object"]
            except KeyError:
                raise KeyError(f"no loss object is present in objective {o}")

            try:
                train_mean = objectives_dict[o]["loss"]["train_mean"]
            except KeyError:
                raise KeyError(f"no train_mean is present in objective {o}")

            try:
                val_mean = objectives_dict[o]["loss"]["val_mean"]
            except KeyError:
                raise KeyError(f"no val_mean is present in objective {o}")

            try:
                in_conf = objectives_dict[o]["in_config"]
            except NotImplementedError:
                raise NotImplementedError(
                    f"no options present in {objectives_dict[o]['in_config']}"
                )

            if in_to_optimizer:
                if not in_to_optimizer == in_conf:
                    raise ValueError(
                        f"The in to optimizer is {in_to_optimizer} but the in_conf for {o} is {in_conf}, they should be the same"
                    )
            else:
                in_to_optimizer = in_conf

            # add loss object to a list for grouping
            loss_names_to_optimize.append(o)
            loss_objs_to_optimize.append(loss_object)
            train_means.append(train_mean)
            val_means.append(val_mean)

        # TODO: this may not be true -- there may only be one loss to optimize
        # create and include joint metric
        joint_name = "__".join(loss_names_to_optimize) + "__joint"
        train_name = joint_name + "_train"
        val_name = joint_name + "_val"
        joint_object_train = tf.keras.metrics.Mean(name=train_name, dtype=tf.float32)
        joint_object_val = tf.keras.metrics.Mean(name=val_name, dtype=tf.float32)

        optimizer_to_loss_name_map[cur_optimizer_name] = {
            "losses_to_optimize": {
                "names": loss_names_to_optimize,
                "objects": loss_objs_to_optimize,
                "record": {"train": {"mean": train_means}, "val": {"mean": val_means}},
                "joint_name": train_name,
                "joint_record": {
                    "train": {"mean": joint_object_train},
                    "val": {"mean": joint_object_val},
                },
            },
            "in_conf": in_conf,
        }

    # ensure all losses are mapped to an optimizer
    obj_not_used = []
    for objective_name, obj_dict in objectives_dict.items():
        # only add objective if it contains a loss
        try:
            _ = obj_dict["loss"]
            if objective_name not in objectives_used:
                obj_not_used.append(objective_name)
        except KeyError:
            pass
    if obj_not_used:
        raise ValueError(f"objectives {obj_not_used} are not mapped to an optimizer")

    return optimizer_to_loss_name_map


def map_in_config_to_objective(objectives_dict):
    in_hash_to_objectives = {}
    for o, d in objectives_dict.items():
        in_conf = d["in_config"]
        in_conf_hash = make_hash(in_conf)
        try:
            stored_conf = in_hash_to_objectives[in_conf_hash]["in_config"]
            if not stored_conf == in_conf:
                raise ValueError(
                    f"the hash is the same, but the in config is different..."
                )
        except KeyError:
            in_hash_to_objectives[in_conf_hash] = {"in_config": in_conf}

        # ? is there a case where there is no objective?
        try:
            stored_objectives = in_hash_to_objectives[in_conf_hash]["objectives"]
            stored_objectives.append(o)
        except KeyError:
            in_hash_to_objectives[in_conf_hash]["objectives"] = [o]

    return in_hash_to_objectives


def create_grouped_metrics(objectives_dict, in_hash_to_objectives):
    in_hash_to_metrics_config = {}

    # loop the different in/out combinations and build metrics for each
    # this dict may become a bit messy because there is the train+val to keep
    # track of
    for k, v in in_hash_to_objectives.items():
        in_hash_to_metrics_config[k] = {"in_config": v["in_config"]}
        in_hash_to_metrics_config[k]["metric_order"] = []
        in_hash_to_metrics_config[k]["objects"] = {"train": [], "val": []}
        for objective in v["objectives"]:
            obj_dict = objectives_dict[objective]
            try:
                cur_metrics = obj_dict["metrics"]
            except KeyError:
                cur_metrics = None

            if cur_metrics:
                in_hash_to_metrics_config[k]["metric_order"].extend(
                    obj_dict["metrics"]["metric_order"]
                )
                in_hash_to_metrics_config[k]["objects"]["train"].extend(
                    obj_dict["metrics"]["train_metrics"]
                )
                in_hash_to_metrics_config[k]["objects"]["val"].extend(
                    obj_dict["metrics"]["val_metrics"]
                )

    return in_hash_to_metrics_config
