from yeahml.config.graph_analysis.build_graph_dict import build_empty_graph
from toposort import toposort, toposort_flatten


def static_analysis_v2(config_dict: dict) -> dict:
    ## build emtpy graph
    graph_dict = build_empty_graph(config_dict)

    ## build dependency dict
    dep_dict = {}
    for _, gnode in graph_dict.items():
        cur_layer_name = gnode.name
        cur_layer_input_deps = gnode.in_name
        dep_dict[cur_layer_name] = set(cur_layer_input_deps)

    # toposort (https://pypi.org/project/toposort/) is used to order the layer
    # dependencies as well as check for circular dependencies
    graph_dependencies = list(toposort(dep_dict))

    return graph_dict, graph_dependencies
