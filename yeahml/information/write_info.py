import json
import pathlib
from typing import Any, Dict


def write_build_information(model_cdict: Dict[str, Any]) -> bool:
    json_path = pathlib.Path(model_cdict["model_root_dir"]).joinpath("info.json")

    data_to_write = {}
    KEYS_TO_WRITE = ["model_hash"]

    if pathlib.Path(json_path).exists():
        with open(json_path) as json_file:
            data = json.load(json_file)

        for k in KEYS_TO_WRITE:
            assert (
                data[k] == model_cdict[k]
            ), f"info at {json_path} already contains the same values for keys {k}, but {json_path}={data[k]} and model config = {model_cdict[k]}\n > possible solution: change the name of the current model?"

    for k in KEYS_TO_WRITE:
        data_to_write[k] = model_cdict[k]
    with open(json_path, "w") as outfile:
        json.dump(data_to_write, outfile)

    return True
