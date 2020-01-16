import pathlib
from typing import Any, Dict

import tensorflow as tf

from yeahml.dataset.handle_data import return_batched_iter


def _apply_ds_hyperparams(hp_cdict: Dict[str, Any], dataset: Any) -> Any:
    # these are hyperparameters applied to the (already created) dataset. Right
    # now only `batch` is supported, but an additional shuffle could be
    # suppported in the future as well as a check as to whether a <method> has
    # previously been applied.

    try:
        batch_size = hp_cdict["dataset"]["batch"]
    except KeyError:
        batch_size = 1
    dataset = dataset.batch(batch_size)

    return dataset


def _get_dataset_from_tfrs(
    ds_type: str, data_cdict: Dict[str, Any], hp_cdict: Dict[str, Any]
) -> Any:
    try:
        dir_path = data_cdict[f"TFR_{ds_type}"]
    except KeyError:
        raise KeyError(f"TFR_{ds_type} option does not exist in {data_cdict.keys()}")

    tfr_train_path: Any = pathlib.Path(data_cdict["TFR_dir"]).joinpath(dir_path)
    dataset: Any = return_batched_iter("train", data_cdict, hp_cdict, tfr_train_path)

    return dataset


def get_configured_dataset(
    data_cdict: Dict[str, Any],
    hp_cdict: Dict[str, Any],
    ds: Any = None,
    ds_type: str = "",
) -> Any:
    if not ds:
        if not ds_type:
            raise ValueError(
                f"please either specify the dataset type (ds_type) or provide a dataset (ds)"
            )
        dataset = _get_dataset_from_tfrs(ds_type, data_cdict, hp_cdict)
    else:
        assert isinstance(
            ds, tf.data.Dataset
        ), f"a {type(ds)} was passed as a training dataset, please pass an instance of {tf.data.Dataset}"
        dataset = ds

    configured_dataset = _apply_ds_hyperparams(hp_cdict, dataset)

    return configured_dataset
