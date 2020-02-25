#!/usr/bin/env python

from yeahml.config.data.parse_data import format_data_config
from yeahml.config.default.default_config import DEFAULT_CONFIG
from yeahml.config.helper import extract_dict_from_path, get_raw_dict_from_string
from yeahml.config.hyper_parameters.parse_hyper_parameters import (
    format_hyper_parameters_config,
)
from yeahml.config.logging.parse_logging import format_logging_config
from yeahml.config.meta.parse_meta import format_meta_config
from yeahml.config.model.parse_model import format_model_config
from yeahml.config.performance.parse_performance import format_performance_config

# components

## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

# TODO: A config logger should be generated / used

# TODO: I don't like this global, but I'm not sure where it belongs yet
# NOTE: it is required that meta be created before model. this may need to change
CONFIG_KEYS = ["meta", "logging", "performance", "data", "hyper_parameters", "model"]


def maybe_extract_from_path(cur_dict: dict) -> dict:
    try:
        cur_path = cur_dict["path"]
        if len(cur_dict.keys()) > 1:
            raise ValueError(
                f"the current dict has a path specified, but also contains other top level keys ({cur_dict.keys()}). please move these keys to path location or remove"
            )
        cur_dict = extract_dict_from_path(cur_path)
    except KeyError:
        pass
    return cur_dict


def primary_config(main_path: str) -> dict:
    main_config_raw = get_raw_dict_from_string(main_path)
    cur_keys = main_config_raw.keys()
    invalid_keys = []
    for key in CONFIG_KEYS:
        if key not in cur_keys:
            invalid_keys.append(key)
            # not all of these *need* to be present, but for now that will be enforced
    if invalid_keys:
        raise ValueError(
            f"The main config does not contain the key(s) {invalid_keys}: current keys: {cur_keys}"
        )

    # build dict containing configs
    config_dict = {}
    for config_type in CONFIG_KEYS:
        raw_config = main_config_raw[config_type]
        # if the main key has a "path" key, then extract from that path
        raw_config = maybe_extract_from_path(raw_config)
        if config_type == "meta":
            formatted_config = format_meta_config(raw_config, DEFAULT_CONFIG["meta"])
        elif config_type == "logging":
            formatted_config = format_logging_config(
                raw_config, DEFAULT_CONFIG["logging"]
            )
        elif config_type == "performance":
            formatted_config = format_performance_config(
                raw_config, DEFAULT_CONFIG["performance"]
            )
        elif config_type == "data":
            formatted_config = format_data_config(raw_config, DEFAULT_CONFIG["data"])
        elif config_type == "hyper_parameters":
            formatted_config = format_hyper_parameters_config(
                raw_config, DEFAULT_CONFIG["hyper_parameters"]
            )
        elif config_type == "model":
            # formatted_config = format_model_config(raw_config, config_dict["meta"])
            formatted_config = format_data_config(raw_config, DEFAULT_CONFIG["model"])
        else:
            raise ValueError(f"config type {config_type} is not yet implemented")
        config_dict[config_type] = formatted_config

    return config_dict


class NOTDEFINED:
    def __init__(self):
        pass


class RAW:
    def __init__(self):
        pass


class g_node:
    def __init__(
        self,
        name=NOTDEFINED,
        source=NOTDEFINED,
        in_name=NOTDEFINED,
        in_source=NOTDEFINED,
        out_name=[],
    ):
        self.name = name
        self.source = source
        self.in_name = in_name
        self.in_source = in_source
        self.out_name = out_name

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.conf_dict


LOOP_ORDER = [("data", ["data", "in"]), ("model", ["model", "layers"])]


def _extract_raw_nodes(source, cur_config):
    cur_dict = {}
    for k, d in cur_config.items():
        cur_dict[k] = g_node(name=k, source=source)
    return cur_dict


def _obtain_nested_dict(nested, outter_dict) -> dict:
    cur_config = None
    for i, v in enumerate(nested):
        if i == 0:
            cur_config = outter_dict[v]
        else:
            cur_config = cur_config[v]
    return cur_config


def _build_skeleton(config_dict):
    # build skeleton graph

    graph_dict = {}
    for name, nested in LOOP_ORDER:
        raw_node_config = _obtain_nested_dict(nested, config_dict)
        empty_node_dict = _extract_raw_nodes(name, raw_node_config)
        graph_dict[name] = empty_node_dict

    # data
    # data_dict = {}
    # for k, d in config_dict["data"]["in"].items():
    #     # k = name of data, d = config spec [shape, dtype]
    #     data_dict[k] = g_node(name=k, in_name=RAW, in_source=RAW)
    # graph_dict["data"] = data_dict

    # preprocessing

    # layers
    # layers_dict = {}
    # for k, d in config_dict["model"]["layers"].items():
    #     # k = name of layer
    #     d_in = d["layer_in_name"]
    #     layers_dict[k] = g_node(name=k)
    # graph_dict["layers"] = data_dict

    return graph_dict


# def _validate_inputs(graph_dict: dict, config_dict: dict):
#     for name, _ in LOOP_ORDER:
#         cur_dict = graph_dict[name]
#         for name, node in cur_dict:


def static_analysis(config_dict: dict) -> dict:
    # There's a lot that could be done here.. but for now, I think just a check
    # to ensure inputs/outputs are specified

    # build dictionary of all nodes in graph
    graph_dict = _build_skeleton(config_dict)

    # validate that all input layers are accounted for
    # graph_dict = graph_dict()

    # loop layers
    # for l_name, d in graph_dict["layers"].items():
    #     in_name = d["in"]

    # for l_name, d in graph_dict["layers"].items():
    #     if issubclass(d["in"], RAW):
    #         pass
    #     elif issubclass(d["in"], NOTDEFINED):
    #         pass

    print(graph_dict)

    return {}


def create_configs(main_path: str) -> dict:

    # parse individual configs
    config_dict = primary_config(main_path)

    # validate graph
    static_dict = static_analysis(config_dict)
    config_dict["static"] = static_dict

    return config_dict
