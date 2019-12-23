import os
from typing import Any, Dict, Tuple

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords


def get_datasets_from_tfrs(
    data_cdict: Dict[str, Any], hp_cdict: Dict[str, Any]
) -> Tuple[Any, Any]:
    tfr_train_path = os.path.join(data_cdict["TFR_dir"], data_cdict["TFR_train"])
    train_ds: Any = return_batched_iter("train", data_cdict, hp_cdict, tfr_train_path)

    tfr_val_path = os.path.join(data_cdict["TFR_dir"], data_cdict["TFR_val"])
    val_ds: Any = return_batched_iter("train", data_cdict, hp_cdict, tfr_val_path)
    return (train_ds, val_ds)
