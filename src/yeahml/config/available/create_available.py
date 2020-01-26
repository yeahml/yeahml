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
    write_dict["meta"]["version"]["yeahml"] = str(yml.__version__)

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


def _obtain_from_class(
    cur_class: Any, subclass: Any = None, exclude_list: List[str] = []
) -> List[str]:
    available_components = []
    available_dict = cur_class.__dict__
    for component_name, component_func in available_dict.items():
        component_name = component_name.lower()
        if inspect.isclass(component_func):
            if subclass:
                if issubclass(component_func, subclass):
                    if component_name not in exclude_list:
                        available_components.append(component_name)
            else:
                if component_name not in exclude_list:
                    available_components.append(component_name)

    return available_components


def _obtain_from_callable(cur_callable: Any, exclude_list: List[str] = []) -> List[str]:
    available_components = []
    available_dict = cur_callable.__dict__
    for component_name, component_func in available_dict.items():
        component_name = component_name.lower()
        if callable(component_func):
            if component_name not in exclude_list:
                available_components.append(component_name)

    return available_components


def write_available_layers(base_dir: Any) -> None:
    available_components = _obtain_from_class(tf.keras.layers, tf.keras.layers.Layer)
    write_dict = _return_write_dict(data=available_components)
    _persist_json(write_dict, base_dir, "layers")


def write_available_activations(base_dir: Any) -> None:
    EXCLUDE_LIST = ["serialize", "deserialize", "get"]
    available_components = _obtain_from_callable(tf.keras.activations, EXCLUDE_LIST)
    write_dict = _return_write_dict(data=available_components)
    _persist_json(write_dict, base_dir, "activations")


def write_available_optimizers(base_dir: Any) -> None:
    available_components = _obtain_from_class(tf.keras.optimizers)
    write_dict = _return_write_dict(data=available_components)
    _persist_json(write_dict, base_dir, "optimizers")


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

    # optimizer
    write_available_optimizers(base_dir)
