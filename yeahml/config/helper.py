import json
import os
import pathlib
import shutil
from io import StringIO

import yaml


def create_exp_dir(root_dir: str, wipe_dirs: bool):
    if wipe_dirs:
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)

    if not os.path.exists(root_dir):
        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True)


def create_standard_dirs(root_dir: str, dirs_to_make: list, wipe_dirs: bool):
    if wipe_dirs:
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
    if not os.path.exists(root_dir):
        pathlib.Path(root_dir).mkdir(parents=True, exist_ok=True)

    new_dirs = {}
    for dir_path in dirs_to_make:
        full_path = os.path.join(root_dir, dir_path)
        if not os.path.exists(full_path):
            pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        new_dirs[dir_path] = full_path

    return new_dirs


def parse_yaml_from_path(path: str) -> dict:
    # return python dict from yaml path
    try:
        with open(path, "r") as stream:
            try:
                y = yaml.load(stream, Loader=yaml.FullLoader)
                return y
            except yaml.YAMLError as exc:
                print(exc)
                return dict()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error > Exiting: the configuration file {path} was not found"
        )


def parse_json_from_path(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as data_file:
            try:
                data = json.loads(data_file.read())
                return data
            except:
                print(f"Error loading json to dict for file {path}")
                return dict()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error > Exiting: the configuration file {path} was not found"
        )


def is_valid_path(cur_string: str) -> bool:
    if os.path.exists(cur_string):
        return True
    else:
        return False


def is_yaml(cur_string: str) -> bool:
    if "\n" in cur_string or "\r" in cur_string:
        return True
    else:
        return False


# TODO: will need to implement json logic as well
def parse_yaml_string(ys):
    fd = StringIO(ys)
    dct = yaml.load(fd, Loader=yaml.FullLoader)
    return dct


def get_raw_dict_from_string(cur_string: str) -> dict:
    if is_yaml(cur_string):
        raw_dict = parse_yaml_string(cur_string)
    elif is_valid_path(cur_string):
        raw_dict = extract_dict_from_path(cur_string)
    else:
        raise ValueError(
            f"The requested string is not a valid configuration file path or yaml string {cur_string}"
        )
    return raw_dict


def extract_dict_from_path(cur_path):
    if cur_path.endswith("yaml") or cur_path.endswith("yml"):
        main_config_raw = parse_yaml_from_path(cur_path)
    elif cur_path.endswith("json"):
        main_config_raw = parse_json_from_path(cur_path)
    if not main_config_raw:
        raise ValueError(
            f"Error > Exiting: the model config file was found {cur_path}, but appears to be empty. It is also possible the file is not valid"
        )
    return main_config_raw
