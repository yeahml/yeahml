import datetime
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

import yeahml as yml
from yeahml.config.available.config import COMPONENT_DICT

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
    comp_class: Any, comp_subclass: Any = None, exclude_list: List[str] = []
) -> List[str]:

    # NOTE: if two components are name Xxxx and xxxx, only one will be included
    # as the name is converted to lower(). This assumes that the cases point to
    # the same underlying implementation (which may not be true) -- this is ok
    # for now (initializers are a good example of this and the assumption holds,
    # but this logic may need to be improved/modified in the future.)

    available_components: List[str] = []
    available_dict = comp_class.__dict__
    for component_name, component_func in available_dict.items():
        component_name = component_name.lower()
        if not component_name.startswith("_") and not component_name.endswith("_"):
            if inspect.isclass(component_func):
                if comp_subclass:
                    if issubclass(component_func, comp_subclass):
                        if component_name not in exclude_list:
                            if component_name not in available_components:
                                available_components.append(component_name)
                else:
                    if component_name not in exclude_list:
                        if component_name not in available_components:
                            available_components.append(component_name)
            elif inspect.isclass(type(component_func)):
                if comp_subclass:
                    if issubclass(type(component_func), comp_subclass):
                        if component_name not in exclude_list:
                            if component_name not in available_components:
                                available_components.append(component_name)
                else:
                    if component_name not in exclude_list:
                        if component_name not in available_components:
                            available_components.append(component_name)
            else:
                pass

    return available_components


def _obtain_from_callable(cur_callable: Any, exclude_list: List[str] = []) -> List[str]:

    # NOTE: if two components are name Xxxx and xxxx, only one will be included
    # as the name is converted to lower(). This assumes that the cases point to
    # the same underlying implementation (which may not be true) -- this is ok
    # for now (initializers are a good example of this and the assumption holds,
    # but this logic may need to be improved/modified in the future.)

    available_components: List[str] = []
    available_dict = cur_callable.__dict__
    for component_name, component_func in available_dict.items():
        component_name = component_name.lower()
        if not component_name.startswith("_") and not component_name.endswith("_"):
            if callable(component_func) and not inspect.isclass(component_func):
                if component_name not in exclude_list:
                    if component_name not in available_components:
                        available_components.append(component_name)

    return available_components


def write_available_component(
    base_dir: Any, component_name: str, component_dict: Dict[str, Any]
) -> None:

    comp_type_dict = component_dict["type"]
    comp_type = component_dict["type"]["name"]
    comp_options = component_dict["type"]["options"]

    try:
        exclude_list = comp_options["exclude_list"]
    except KeyError:
        exclude_list = []
    # remove the generic case
    exclude_list.append(component_name)

    if comp_type == "class":
        comp_class = comp_options["class"]
        try:
            comp_subclass = comp_options["subclass"]
        except KeyError:
            comp_subclass = None
        available_components = _obtain_from_class(
            comp_class, comp_subclass, exclude_list
        )
    elif comp_type == "callable":
        comp_callable = comp_options["callable"]
        available_components = _obtain_from_callable(comp_callable, exclude_list)
    else:
        raise ValueError(
            f"In {component_name} the :type:name {comp_type_dict['name']} is not supported, please specify one of ['class', 'callable'] (component_dict)"
        )

    write_dict = _return_write_dict(data=available_components)
    _persist_json(write_dict, base_dir, f"{component_name}")


if __name__ == "__main__":
    ROOT_YML_DIR = Path(yml.__file__).parent
    base_dir = (
        ROOT_YML_DIR.joinpath("config").joinpath("available").joinpath("components")
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    for comp_name, comp_dict in COMPONENT_DICT.items():
        write_available_component(base_dir, comp_name, comp_dict)
