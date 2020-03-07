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
            formatted_config = format_model_config(raw_config, DEFAULT_CONFIG["model"])
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
        config_location=NOTDEFINED,
        in_name=None,
        in_source=None,
        out_name=None,
        out_source=None,
        startpoint=None,
        endpoint=None,
        label=None,
    ):
        self.name = name
        self.config_location = config_location
        self.in_name = in_name
        self.in_source = in_source
        self.out_name = out_name
        self.out_source = out_source
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.label = label

    def __str__(self):
        return str(self.__class__) + f" @ {hex(id(self))}" + ": " + str(self.__dict__)

    def __call__(self):
        return self.conf_dict


LOOP_ORDER = [("data", ["data", "in"]), ("model", ["model", "layers"])]


def _extract_raw_nodes(config_location, cur_config):
    cur_dict = {}
    for k, d in cur_config.items():

        # only data objects have label attribute
        try:
            is_label = d["label"]
        except KeyError:
            is_label = False

        cur_dict[k] = g_node(
            name=k,
            config_location=config_location,
            startpoint=d["startpoint"],
            endpoint=d["endpoint"],
            label=is_label,
        )
    return cur_dict


def _obtain_nested_dict(nested_keys, outter_dict) -> dict:
    cur_config = None
    for i, v in enumerate(nested_keys):
        if i == 0:
            cur_config = outter_dict[v]
        else:
            cur_config = cur_config[v]
    return cur_config


def _build_empty_graph(config_dict):
    # build skeleton graph, all nodes are present in this graph

    graph_dict = {}
    for outter_config_key, nested_keys in LOOP_ORDER:
        raw_node_config = _obtain_nested_dict(nested_keys, config_dict)
        empty_node_dict = _extract_raw_nodes(outter_config_key, raw_node_config)
        # TODO: ensure there aren't any name overwrites here
        graph_dict = {**graph_dict, **empty_node_dict}

    return graph_dict


def _get_node_by_name(search_name, config_dict):
    for name, nested_keys in LOOP_ORDER:
        cur_conf_dict = _obtain_nested_dict(nested_keys, outter_dict=config_dict)
        if search_name in cur_conf_dict.keys():
            try:
                cur_node = cur_conf_dict[search_name]
            except KeyError:
                cur_node = NOTDEFINED
            return (cur_node, nested_keys[0])
    raise ValueError(f"node {search_name} is not locatable")


def get_config_node_input(node, location):
    if location == "data":
        cur_in = RAW
    else:
        try:
            cur_in = node["layer_in_name"]
        except KeyError:
            raise ValueError(f"'layer_in_name' is not defined for {node} in {location}")
    return cur_in


def _validate_inputs(config_dict: dict, graph_dict: dict):

    # loop the graph dict
    for node_name, node in graph_dict.items():
        assert (
            node_name == node.name
        ), f"node_name ({node_name}) != node.name ({node.name})"

        # NOTE: should the data spec be parsed to include information about
        # where the data is coming from (from the raw source - e.g.
        # col_name, ...)
        config_node, location = _get_node_by_name(node_name, config_dict)

        # TODO: convert to named tuple?
        locs = node.in_source
        if locs:
            new_locs = locs + location
        else:
            new_locs = [location]
        node.in_source = new_locs

        in_node_name = get_config_node_input(config_node, location)
        in_names = node.in_name
        if in_names:
            new_in_names = in_names + in_node_name
        else:
            if isinstance(in_node_name, list):
                new_in_names = in_node_name
            else:
                new_in_names = [in_node_name]
        node.in_name = new_in_names

    return True


def build_chain(call_chain, node, graph_dict):
    # recursive function to build chain of input to output
    # I think I'm doing this backwards.. maybe a better approach would be to
    # build the chain from back to front and then reverse it?
    if node.startpoint:
        call_chain.append(node.name)
        return call_chain
    else:
        parent_names = node.in_name
        if len(parent_names) > 1:
            parent_chains = []
            for i, parent in enumerate(parent_names):
                # recursively build chain for each parent
                parent_chain = build_chain([], graph_dict[parent], graph_dict)
                parent_chains.append(parent_chain)

            # all parent chains are not contained in parent_chains
            branch_tuple = ((len(parent_names), parent_chains), node.name)
            call_chain.append(branch_tuple)
            call_chain.append(node.name)
        else:
            new_chain = build_chain(call_chain, graph_dict[parent_names[0]], graph_dict)
            if parent_names[0] == new_chain[-1]:
                call_chain.append(node.name)

    return call_chain


def create_subgraphs(config_dict, graph_dict):
    # NOTE: is this right?
    # 1. loop out nodes --> build chains from out to in
    subgraphs = {}
    for node_name, node in graph_dict.items():
        if node.endpoint:
            if not node.label:
                # it's an endpoint, but it's not a label (labels don't need to
                # be built)
                chain = build_chain([], node, graph_dict)
                subgraphs[node.name] = {"sequence": chain}

    return subgraphs


def _extract_paths(d):
    # https://stackoverflow.com/questions/23981553/get-all-values-from-nested-dictionaries-in-python
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, dict):
                yield from _extract_paths(v)
            else:
                yield v


def _extract_all_nodes_from_paths(path_lists):
    """extract all nodes into a set from the list of paths"""
    # there's likely a cleaner way to do this
    nodes = set()
    for v in path_lists:
        if isinstance(v, tuple):
            # tuple = ((number, [node_a, node_b, etc...]), out_name)
            ex_nodes = v[0][1]
            for n in ex_nodes:
                if isinstance(n, list):
                    for nn in n:
                        nodes.add(nn)
                elif isinstance(n, str):
                    nodes.add(n)
                else:
                    raise ValueError(
                        f"type {type(n)} was found in the path list from the tuple"
                    )
        elif isinstance(v, str):
            nodes.add(v)
        else:
            raise ValueError(f"type {type(v)} was found in the path list")
    return nodes


def static_analysis(config_dict: dict) -> dict:
    # There's a lot that could be done here.. but for now, I think just a check
    # to ensure inputs are specified

    # NOTE: this will undoubtedly need to be optimized. there is a lot of looping
    # going on here

    # TODO: use startpoint/endpoint logic

    # build dictionary of all nodes in graph
    graph_dict = _build_empty_graph(config_dict)

    # validate that all input layers are accounted for
    valid_inputs = _validate_inputs(config_dict, graph_dict)

    # could loop for NOTDEFINED here
    # TODO: Analyze graph_dict to see if there are any "dead_ends"
    subgraphs = create_subgraphs(config_dict, graph_dict)

    # exhaust generator and convert, extract list [[path_lists]]
    path_lists = list(_extract_paths(subgraphs))[0]
    # ensure all values in the graph dict appear in the paths
    nodes_in_path = _extract_all_nodes_from_paths(path_lists)
    for n, nd in graph_dict.items():
        if not nd.label:
            # labels don't need to be checked since they aren't built
            if n not in nodes_in_path:
                raise ValueError(
                    f"node {n} does not appear in any of the paths from inputs to outputs"
                )

    # could detect cycles here
    # for k,_ in graph_dict:

    # TODO: depending on the implementation, we may need to create multiple models
    # based on the `graph_dict` components. This is also likely where we should log
    # the graph information/structure ?

    return graph_dict, subgraphs


def create_configs(main_path: str) -> dict:

    # parse individual configs
    config_dict = primary_config(main_path)

    # build the order of inputs into the model. This logic will likely need to
    # change as inputs become more complex
    input_order = []
    for name, config in config_dict["data"]["in"].items():
        if config["startpoint"]:
            if not config["label"]:
                input_order.append(name)
    if not input_order:
        raise ValueError("no inputs have been specified to the model")

    # loop model to ensure all outputs are accounted for
    output_order = []
    for name, config in config_dict["model"]["layers"].items():
        if config["endpoint"]:
            output_order.append(name)
    if not output_order:
        raise ValueError("no outputs have been specified for the model")
    config_dict["model_io"] = {"inputs": input_order, "outputs": output_order}

    # validate graph
    static_dict, subgraphs = static_analysis(config_dict)
    config_dict["static"] = static_dict
    config_dict["subgraphs"] = subgraphs

    return config_dict
