import datetime
import inspect
import json
from pathlib import Path
from typing import List

import tensorflow as tf

import yeahml as yml

# NOTE: I'm not sure this is the way I want to approach this.


ROOT_YML_DIR = Path(yml.__file__).parent
CUR_AVAIL = ROOT_YML_DIR.joinpath("config").joinpath("available")


def write_available_layers() -> None:
    layer_path = CUR_AVAIL.joinpath("layers.json")
    write_dict = {}

    def _return_available_layers() -> List[str]:
        # logic to get all layers in a class
        available_layers = []
        available_keras_layers = tf.keras.layers.__dict__
        for layer_name, layer_func in available_keras_layers.items():
            if inspect.isclass(layer_func):
                if issubclass(layer_func, tf.keras.layers.Layer):
                    available_layers.append(layer_name.lower())

        return available_layers

    write_dict["meta"] = {}
    write_dict["meta"]["update_time_str"] = str(datetime.datetime.now())
    write_dict["data"] = _return_available_layers()

    with open(layer_path, "w") as fp:
        json.dump(write_dict, fp, indent=2)


if __name__ == "__main__":
    write_available_layers()
