from typing import Any, Dict, List

import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metric import configure_metric

# TODO: I'm not sure where this belongs yet
ACCEPTED_LOSS_TRACK = {"mean": tf.keras.metrics.Mean}


def _get_metrics(
    metric_config: Dict[str, Any], dataset_names: List[str], objective_name: str
) -> Dict[str, Dict[str, Any]]:
    """[summary]
    
    Parameters
    ----------
    metric_config : Dict[str, Any]
        e.g.
            {'type': ['meansquarederror'], 'options': [None]}
    dataset_names : List[str]
        e.g.
            ['train', 'val']
    objective_name : str
        used for providing a name for the metric (if not provided by the user)
        e.g.
            main_obj
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        {
            "meansquarederror": {
                "train": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f2bb443fd30>",
                "val": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f2bb4420f98>",
        }}
    """

    assert len(metric_config["options"]) == len(
        metric_config["type"]
    ), f"len of options does not len of metrics: {len(metric_config['options'])} != {len(metric_config['type'])}"

    # loop operations and options
    metric_ds_to_metric = {}
    try:
        met_opts = metric_config["options"]
    except KeyError:
        met_opts = None
    for i, metric_name in enumerate(metric_config["type"]):
        metric_ds_to_metric[metric_name] = {}
        if met_opts:
            met_opt_dict = met_opts[i]
        else:
            met_opt_dict = {}

        if met_opt_dict:
            try:
                _ = met_opt_dict["name"]
            except KeyError:
                met_opt_dict[
                    "name"
                ] = f"metric_{objective_name}_{metric_name}_{dataset_name}"

        for dataset_name in dataset_names:
            metric_fn = configure_metric(metric_name, met_opt_dict)
            metric_ds_to_metric[metric_name][dataset_name] = metric_fn

    return metric_ds_to_metric


def _get_loss(
    loss_config: Dict[str, Any], dataset_names: List[str], objective_name: str
) -> Dict[str, Any]:
    """Create a dictionary containing the loss and corresponding tracking object for each datasets
    
    Parameters
    ----------
    loss_config : Dict[str, Any]
        The type of loss, the options for configuring the loss object, and a
        list of the type of tracking to perform on the given loss function
        e.g.
            {'type': 'mse', 'options': None, 'track': ['mean']}
    dataset_names : List[str]
        e.g.
            ['train', 'val']
    objective_name : str
        used for providing a name for the metric (if not provided by the user)
        e.g.
            main_obj
    
    Returns
    -------
    Dict[str, Any]
        e.g.
            {"object": "<function mean_squared_error at 0x7f2bd628d268>",
                "track": {
                    "train": {
                        "mse": {
                            "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f2bb4420748>"
                        }
                    },
                    "val": {
                        "mse": {
                            "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f2bb4420860>"
            }},},}
    
    Raises
    ------
    KeyError
        [description]
    """

    loss_ds_to_loss = {}

    # build tf loss object
    loss_object = configure_loss(loss_config)
    loss_ds_to_loss["object"] = loss_object

    try:
        loss_track = loss_config["track"]
        loss_ds_to_loss["track"] = {}
    except KeyError:
        loss_track = None

    if loss_track:
        for name in loss_track:
            try:
                tf_track_class = ACCEPTED_LOSS_TRACK[name]
            except KeyError:
                raise KeyError(
                    f"{name} is not an accepted track method please select from {ACCEPTED_LOSS_TRACK.keys()}"
                )

            for dataset_name in dataset_names:
                loss_ds_to_loss["track"][dataset_name] = {}
                loss_ds_to_loss["track"][dataset_name][loss_config["type"]] = {}
                # TODO: I'm not consistent with naming here.. should the user be
                # allowed to name this?
                tf_tracker_name = (
                    f"loss_{objective_name}_{loss_config['type']}_{name}_{dataset_name}"
                )
                dtype = tf.float32
                tf_tracker = tf_track_class(name=tf_tracker_name, dtype=dtype)
                loss_ds_to_loss["track"][dataset_name][loss_config["type"]][
                    name
                ] = tf_tracker

    return loss_ds_to_loss


def get_objectives(
    objectives: Dict[str, Any], dataset_names: List[str]
) -> Dict[str, Any]:
    """Builds the objective configuration for a given objective
    
    Parameters
    ----------
    objectives : Dict[str, Any]
        e.g.
            {
                "main_obj": {
                    "loss": {"type": "mse", "options": None, "track": ["mean"]},
                    "metric": {"type": ["meansquarederror"], "options": [None]},
                    "in_config": {
                        "type": "supervised",
                        "options": {"prediction": "dense_out", "target": "target_v"},
                    },},
                ...}
    dataset_names : List[str]
        [description]
        e.g.
            ['train', 'val']
    
    Returns
    -------
    Dict[str, Any]
        dictionary containing the corresponding in_config, loss, and metric for
        a given objective
        e.g.
            {
                "main_obj": {
                    "in_config": {
                        "type": "supervised",
                        "options": {"prediction": "dense_out", "target": "target_v"},
                    },
                    "loss": {
                        "object": "<function mean_squared_error at 0x7f2bd628d268>",
                        "track": {
                            "train": {
                                "mse": {
                                    "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f2bb42d2f98>"
                            }},
                            "val": {
                                "mse": {
                                    "mean": "<tensorflow.python.keras.metrics.Mean object at 0x7f2bb42d2f28>"
                    }},},},
                    "metrics": {
                        "meansquarederror": {
                            "train": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f2bb42e1668>",
                            "val": "<tensorflow.python.keras.metrics.MeanSquaredError object at 0x7f2bb42e1ba8>",
                ...
            }},}}
    
    Raises
    ------
    ValueError
        [description]
    ValueError
        [description]
    ValueError
        [description]
    NotImplementedError
        [description]
    """

    if not isinstance(dataset_names, list):
        if not isinstance(dataset_names, str):
            raise ValueError(
                f"dataset_names ({dataset_names}) must be of type list of strings or string not {type(dataset_names)}"
            )
    else:
        for o in dataset_names:
            if not isinstance(o, str):
                raise ValueError(
                    f"object ({o}) in ({dataset_names}) must be of type string not {type(o)}"
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
            loss_ds_to_loss = _get_loss(loss_config, dataset_names, objective_name)

        else:
            loss_ds_to_loss = None

        if metric_config:
            metric_ds_to_metric = _get_metrics(
                metric_config, dataset_names, objective_name
            )
        else:
            metric_ds_to_metric = None

        obj_conf[objective_name] = {
            "in_config": in_config,
            "loss": loss_ds_to_loss,
            "metrics": metric_ds_to_metric,
        }

    # TODO: eventually this needs to be removed/changed to a more general check.
    # Currently, only supervised is accepted
    for objective_name, obj_dict in obj_conf.items():
        if obj_dict["in_config"]["type"] != "supervised":
            raise NotImplementedError(
                f"only 'supervised' is accepted as the type for the in_config of {objective_name}, not {obj_conf['in_config']['type']} yet..."
            )

    return obj_conf
