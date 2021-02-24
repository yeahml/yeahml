from typing import Any, Dict, List

import tensorflow as tf

from yeahml.build.components.loss import configure_loss
from yeahml.build.components.metric import configure_metric

# TODO: I'm not sure where this belongs yet
ACCEPTED_LOSS_TRACK = {"mean": tf.keras.metrics.Mean}


def _get_metrics(
    metric_config: Dict[str, Any], datasplit_names: List[str], objective_name: str
) -> Dict[str, Dict[str, Any]]:
    """[summary]
    
    Parameters
    ----------
    metric_config : Dict[str, Any]
        e.g.
            {'type': ['meansquarederror'], 'options': [None]}
    datasplit_names : List[str]
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

        for datasplit_name in datasplit_names:
            if met_opt_dict:
                try:
                    _ = met_opt_dict["name"]
                except KeyError:
                    met_opt_dict[
                        "name"
                    ] = f"metric_{objective_name}_{metric_name}_{datasplit_name}"
            metric_fn = configure_metric(metric_name, met_opt_dict)
            metric_ds_to_metric[metric_name][datasplit_name] = metric_fn

    return metric_ds_to_metric


def _get_loss(
    loss_config: Dict[str, Any], datasplit_names: List[str], objective_name: str
) -> Dict[str, Any]:
    """Create a dictionary containing the loss and corresponding tracking object for each datasets
    
    Parameters
    ----------
    loss_config : Dict[str, Any]
        The type of loss, the options for configuring the loss object, and a
        list of the type of tracking to perform on the given loss function
        e.g.
            {'type': 'mse', 'options': None, 'track': ['mean']}
    datasplit_names : List[str]
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

    if isinstance(loss_track, str):
        loss_track = [loss_track]

    if loss_track:
        for name in loss_track:
            try:
                tf_track_class = ACCEPTED_LOSS_TRACK[name]
            except KeyError:
                raise KeyError(
                    f"{name} is not an accepted track method please select from {ACCEPTED_LOSS_TRACK.keys()}"
                )

            for datasplit_name in datasplit_names:
                loss_ds_to_loss["track"][datasplit_name] = {}
                loss_ds_to_loss["track"][datasplit_name][loss_config["type"]] = {}
                # TODO: I'm not consistent with naming here.. should the user be
                # allowed to name this?
                tf_tracker_name = f"loss_{objective_name}_{loss_config['type']}_{name}_{datasplit_name}"
                dtype = tf.float32
                tf_tracker = tf_track_class(name=tf_tracker_name, dtype=dtype)
                loss_ds_to_loss["track"][datasplit_name][loss_config["type"]][
                    name
                ] = tf_tracker

    return loss_ds_to_loss


def get_objectives(
    objectives: Dict[str, Any],
    dataset_dict: Dict[str, Dict[str, Any]],
    target_splits=None,
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
                        'dataset': 'abalone'
                    },},
                ...}
    dataset_dict : Dict[str, Dict[str, Any]]
        [description]
        e.g.
            {
                'abalone': 
                {
                    'train': '<BatchDataset shapes: (None,), types: tf.int32'>, 
                    'val': '<BatchDataset shapes: (None,), types: tf.int32>'
                }}
    
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

    obj_conf = {}
    for objective_name, objective_config in objectives.items():
        # in_config: defines the type (e.g. supervised) and io (e.g. pred, gt)
        # loss --> defines whether a loss is present
        # metric --> defines whether a metric is present
        in_config = objective_config["in_config"]
        ds_name = in_config["dataset"]
        try:
            tf_dataset_dict = dataset_dict[ds_name]
        except KeyError:
            raise KeyError(
                f"no dataset named {ds_name} is included in the dataset_dict {dataset_dict}"
            )

        datasplit_names = list(tf_dataset_dict.keys())
        if target_splits:
            if isinstance(target_splits, list):
                pass
            elif isinstance(target_splits, str):
                target_splits = [target_splits]
            else:
                raise ValueError(f"{target_splits} is not of type list or str")
            assert set(target_splits).issubset(
                set(datasplit_names)
            ), f"{target_splits} is not a subset of {datasplit_names}"
            datasplit_names = target_splits

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
            loss_ds_to_loss = _get_loss(loss_config, datasplit_names, objective_name)
        else:
            loss_ds_to_loss = None

        if metric_config:
            metric_ds_to_metric = _get_metrics(
                metric_config, datasplit_names, objective_name
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
