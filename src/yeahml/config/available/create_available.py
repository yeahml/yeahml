import datetime
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

import yeahml as yml

# NOTE: I'm not sure this is the way I want to approach this.


def _return_write_dict(data: List[str]) -> Dict[str, Any]:

    write_dict: Dict[str, Any] = {}
    write_dict["meta"] = {}
    write_dict["meta"]["update_time_str"] = str(datetime.datetime.now())
    write_dict["meta"]["version"] = {}
    write_dict["meta"]["version"]["tensorflow"] = str(tf.__version__)

    write_dict["data"] = data
    return write_dict


def _persist_json(
    data_dict: Dict[str, Any], base_dir: Any, component_name: str
) -> None:
    if not data_dict:
        raise ValueError("no data passed")

    write_path = base_dir.joinpath(f"{component_name}.json")
    with open(write_path, "w") as fp:
        json.dump(data_dict, fp, indent=2)


def write_available_layers(base_dir: Any) -> None:
    def _return_available_layers() -> List[str]:
        # logic to get all layers in a class
        available_components = []
        available_keras_layers = tf.keras.layers.__dict__
        for layer_name, layer_func in available_keras_layers.items():
            if inspect.isclass(layer_func):
                if issubclass(layer_func, tf.keras.layers.Layer):
                    available_components.append(layer_name.lower())
        return available_components

    write_dict = _return_write_dict(data=_return_available_layers())
    _persist_json(write_dict, base_dir, "layers")


def write_available_activations(base_dir: Any) -> None:
    EXCLUDE_LIST = ["serialize", "deserialize", "get"]

    def _return_available_activations() -> List[str]:
        # logic to get all layers in a class
        available_components = []
        available_keras_layers = tf.keras.activations.__dict__
        for layer_name, layer_func in available_keras_layers.items():
            layer_name = layer_name.lower()
            if callable(layer_func):
                if layer_name not in EXCLUDE_LIST:
                    available_components.append(layer_name.lower())

        return available_components

    write_dict = _return_write_dict(data=_return_available_activations())
    _persist_json(write_dict, base_dir, "activations")


if __name__ == "__main__":
    ROOT_YML_DIR = Path(yml.__file__).parent
    base_dir = (
        ROOT_YML_DIR.joinpath("config").joinpath("available").joinpath("components")
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # layers
    write_available_layers(base_dir)

    # activations
    write_available_activations(base_dir)
