from typing import Any, Dict, List

# `**` denotes that every key in this nested location should be looped through,
# regardless of the name
LOOP_ORDER = [
    ("data", ["data", "datasets", "**", "in"]),
    ("model", ["model", "layers"]),
]


class NOTDEFINED:
    def __init__(self):
        pass


class g_node:
    def __init__(
        self,
        name=NOTDEFINED,
        config_location=NOTDEFINED,
        in_name=None,
        # in_source=None,
        out_name=None,
        out_source=None,
        startpoint=None,
        endpoint=None,
        label=None,
    ):
        self.name = name
        self.config_location = config_location
        self.in_name = in_name
        # self.in_source = in_source
        self.out_name = out_name
        self.out_source = out_source
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.label = label

    def __str__(self):
        return str(self.__class__) + f" @ {hex(id(self))}" + ": " + str(self.__dict__)

    def __call__(self):
        return self.conf_dict


def get_node_config_by_name(search_name, config_dict):
    for name, nested_keys in LOOP_ORDER:
        # LIST
        cur_conf_dicts = _obtain_items_from_nested_dict(
            nested_keys, outter_dict=config_dict
        )

        # assert len(cur_conf_dicts) == 1, f"multiple nodes were returned for {}"
        if search_name in cur_conf_dicts.keys():
            try:
                cur_node_config = cur_conf_dicts[search_name]
            except KeyError:
                cur_node_config = NOTDEFINED
            return cur_node_config
    raise ValueError(f"node {search_name} is not locatable")


def _extract_raw_nodes(cur_configs):

    cur_dict = {}
    for cur_layer_name, cur_config in cur_configs.items():

        object_dict = cur_config["object_dict"]

        # only data objects have label attribute
        try:
            is_label = object_dict["label"]
        except KeyError:
            is_label = False

        # obtain the names of the layers that are dependencies to the current
        # layer and convert to a list
        try:
            layer_in_name = object_dict["layer_in_name"]
        except KeyError:
            layer_in_name = None
        if isinstance(layer_in_name, list):
            layer_in_names = layer_in_name
        else:
            layer_in_names = [layer_in_name]

        cur_dict[cur_layer_name] = g_node(
            name=cur_layer_name,
            config_location=cur_config["source_keys"],
            startpoint=object_dict["startpoint"],
            endpoint=object_dict["endpoint"],
            in_name=layer_in_names,
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
    ret_dict = {}
    for k, d in cur_config.items():
        ret_dict[k] = {"source_keys": nested_keys, "object_dict": d}
    return ret_dict


def _obtain_nested_dataset_dict(nested_keys, outter_dict) -> dict:
    full_dict = {}
    ind = nested_keys.index("**")
    pre_keys = nested_keys[:ind]
    post_keys = nested_keys[ind + 1 :]
    pre_dict = _obtain_nested_dict(pre_keys, outter_dict)

    for k, cur_pre_dict in pre_dict.items():
        cur_inner_dict = _obtain_nested_dict(post_keys, cur_pre_dict["object_dict"])
        src_keys = pre_keys + [k] + post_keys
        for k, d in cur_inner_dict.items():
            full_dict[k] = {"source_keys": src_keys, "object_dict": d["object_dict"]}

    return full_dict


def _obtain_items_from_nested_dict(
    nested_keys: List[str], outter_dict: Dict[str, Any]
) -> dict:
    """Extract the relevant components from a nested dictionary

    if the nested keys obtain a "**", then the dict is looped to obtain all
    values from the keys beyond the "**".
    
    Parameters
    ----------
    nested_keys : List[str]
        e.g.
            ['data', 'datasets', '**', 'in']
    outter_dict : Dict[str, Any]
        e.g.
            {
                "data": {
                    "datasets": {
                        "abalone": {
                            "in": {
                                "feature_a": {
                                    "shape": [2, 1],
                                    "dtype": "float64",
                                    "startpoint": True,
                                    "endpoint": False,
                                    "label": False,
                                },
                                "target_v": {
                                    "shape": [1, 1],
                                    "dtype": "int32",
                                    "startpoint": True,
                                    "endpoint": True,
                                    "label": True,
            },}}}}}

    
    Returns
    -------
    dict
        [description]
        e.g.
            TODO
            
    """

    if "**" in nested_keys:
        cur_config = _obtain_nested_dataset_dict(nested_keys, outter_dict)
    else:
        cur_config = _obtain_nested_dict(nested_keys, outter_dict)

    return cur_config


def build_empty_graph(config_dict):
    # build skeleton graph dictionary, all nodes are present in this graph

    graph_dict = {}
    # (outter_config_key, nested_keys)
    for _, nested_keys in LOOP_ORDER:
        # LIST
        raw_node_configs = _obtain_items_from_nested_dict(nested_keys, config_dict)
        empty_node_dict = _extract_raw_nodes(raw_node_configs)
        # TODO: ensure there aren't any name overwrites here
        graph_dict = {**graph_dict, **empty_node_dict}

    return graph_dict
