import json
import os
from io import StringIO

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class UniqueKeyLoader(Loader):
    # https://gist.github.com/pypt/94d747fe5180851196eb#gistcomment-2084028
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None,
                None,
                "expected a mapping node, but found %s" % node.id,
                node.start_mark,
            )
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unacceptable key (%s)" % exc,
                    key_node.start_mark,
                )
            # check for duplicate keys
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate key",
                    key_node.start_mark,
                    f"duplicate key: {key} -- please rename the key to something else",
                )
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


yaml.SafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, UniqueKeyLoader.construct_mapping
)


def _parse_yaml_from_path(path: str) -> dict:
    # return python dict from yaml path
    try:
        with open(path, "r") as stream:
            try:
                y = yaml.load(stream, Loader=yaml.SafeLoader)
                return y
            except yaml.YAMLError as exc:
                print(exc)
                return dict()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Error > Exiting: the configuration file {path} was not found"
        )


def _parse_json_from_path(path: str) -> dict:
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


def _is_yaml(cur_string: str) -> bool:
    if "\n" in cur_string or "\r" in cur_string:
        return True
    else:
        return False


# TODO: will need to implement json logic as well
def _parse_yaml_string(ys):
    fd = StringIO(ys)
    dct = yaml.load(fd, Loader=yaml.SafeLoader)
    return dct


def get_raw_dict_from_string(cur_string: str) -> dict:
    if _is_yaml(cur_string):
        raw_dict = _parse_yaml_string(cur_string)
    elif os.path.exists(cur_string):
        raw_dict = extract_dict_from_path(cur_string)
    else:
        raise ValueError(
            f"The requested string is not a valid configuration file path or yaml string {cur_string}"
        )
    return raw_dict


def extract_dict_from_path(cur_path):
    if cur_path.endswith("yaml") or cur_path.endswith("yml"):
        main_config_raw = _parse_yaml_from_path(cur_path)
    elif cur_path.endswith("json"):
        main_config_raw = _parse_json_from_path(cur_path)
    if not main_config_raw:
        raise ValueError(
            f"Error > Exiting: the model config file was found {cur_path}, but appears to be empty. It is also possible the file is not valid"
        )
    return main_config_raw
