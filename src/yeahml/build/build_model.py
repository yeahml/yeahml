from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

from yeahml.build.build_layers import build_hidden_block
from yeahml.build.layers.config import NOTPRESENT
from yeahml.information.write_info import write_build_information

# from yeahml.build.get_components import get_logits_and_preds
from yeahml.log.yf_logging import config_logger


# Helper to make the output "consistent"
def reset_graph_deterministic(seed: int = 42) -> None:
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph_deterministic")
    # there is no option for deterministic behavior yet...
    # > tf issue https://github.com/tensorflow/tensorflow/issues/18096
    # os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()
    # np.random.seed(seed)


def reset_graph(seed: int = 42) -> None:
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph")
    # tf.reset_default_graph()
    tf.random.set_seed(seed)
    tf.keras.backend.clear_session()


def _configure_input(cur_name, cur_config):

    # TODO: this section should be redone to match the layer API. but for now
    # I'm going to continue on with this minimal approach
    dtype = cur_config["dtype"]
    shape = cur_config["shape"]
    # print(cur_config)
    if cur_config["startpoint"]:
        out = tf.keras.layers.Input(shape=shape, dtype=dtype, name=cur_name)
    else:
        # all data layers should be a startpoint
        raise ValueError(f"current data layer config:{cur_config} is not a startpoint")

    return out


def _configure_layer(cur_name, cur_config):

    # 'layer_base', 'options', 'layer_in_name'
    layer_base = cur_config["layer_base"]

    # default layer function
    layer_fn = layer_base["func"]

    # assemble layer with user values (or default if not specified)
    # another way to do this could be just overwrite the values the user
    # specifies, but I want the ability to set defaults for each layer
    # eventually (in case we would want to change them)
    layer_args = layer_base["func_args"]
    layer_defaults = layer_base["func_defaults"]
    user_values = cur_config["options"]["user_vals"]

    param_dict = {}
    for i, param_name in enumerate(layer_args):
        user_val = user_values[i]
        if user_val:
            param_dict[param_name] = user_val
        else:
            def_val = layer_defaults[i]
            if isinstance(def_val, NOTPRESENT):
                raise ValueError(
                    f"param {param_name} for layer {cur_name} not specified, but is required"
                )
            else:
                param_dict[param_name] = def_val

                # overwrite name with layer name if not exist
                if param_name == "name":
                    if not def_val:
                        param_dict[param_name] = cur_name
    configured_layer = layer_fn(**param_dict)

    return configured_layer


def build_model(config_dict: Dict[str, Dict[str, Any]]) -> Any:

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]
    static_cdict: Dict[str, Any] = config_dict["static"]
    subgraphs_cdict: Dict[str, Any] = config_dict["subgraphs"]
    model_io_cdict: Dict[str, Any] = config_dict["model_io"]

    full_exp_path = (
        Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
    )
    logger = config_logger(full_exp_path, log_cdict, "build")
    logger.info("-> START building graph")

    try:
        reset_graph_deterministic(meta_cdict["seed"])
    except KeyError:
        reset_graph()

    full_exp_path = (
        Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
    )
    g_logger = config_logger(full_exp_path, log_cdict, "graph")

    # configure/build all layers and save in lookup table
    built_nodes = {}
    for name, node in static_cdict.items():
        if node.config_location == "model":
            blueprint = model_cdict["layers"][node.name]
        elif node.config_location == "data":
            blueprint = data_cdict["in"][node.name]
        else:
            raise ValueError(f"layer {name} can't be found in {node.config_location}")

        if node.startpoint:
            if node.label:
                pass
            else:
                func = None
                out = _configure_input(name, blueprint)
        else:
            func = _configure_layer(name, blueprint)
            out = None

        # TODO: this is a quick fix. the issue is that a node that is a label,
        # does not need to be built as a layer in the graph -- it is only used
        # as a target during training and therefore does not need to be included
        # here
        if not node.label:
            built_nodes[name] = {"out": out, "func": func}

    # connect the subgraphs
    for end_name, subgraph in subgraphs_cdict.items():

        prev_out = None
        prev_out_exists = False
        # the bool flag is needed since: "OperatorNotAllowedInGraphError: using a
        # `tf.Tensor` as a Python `bool` is not allowed in Graph execution. Use
        # Eager execution or decorate this function with @tf.function." meaning,
        # we can't say if prev_out:
        seq = subgraph["sequence"]
        # print(f"{end_name} @ {seq}")
        for cur_name_in_seq in seq:
            # print(f"> {cur_name_in_seq}")

            # obtain
            try:
                cur_built_node = built_nodes[cur_name_in_seq]
            except KeyError:
                # TODO: this message will have to be expanded for huge graphs
                raise KeyError(
                    f"node ({cur_name_in_seq}) from seq {seq} not found in built nodes {built_nodes.keys()}"
                )

            if prev_out_exists:
                try:
                    cur_func = cur_built_node["func"]
                except KeyError:
                    raise KeyError(
                        f"node ({cur_name_in_seq}) function has not been built yet"
                    )
                # make the connection, store the connected node
                out = cur_func(prev_out)
                cur_built_node["out"] = out
                prev_out = out
            else:
                try:
                    prev_out = cur_built_node["out"]
                    prev_out_exists = True
                except KeyError:
                    raise KeyError(
                        f"node ({cur_name_in_seq}) does has not been created yet"
                    )

    model_input_tensors = []
    for name in model_io_cdict["inputs"]:
        try:
            node_d = built_nodes[name]
        except KeyError:
            raise KeyError(f"{name} not found in built nodes when creating inputs")

        try:
            out = node_d["out"]
        except KeyError:
            raise KeyError(
                f"out was not created for {name} when creating tensor inputs"
            )
        model_input_tensors.append(out)
    if not model_input_tensors:
        raise ValueError(f"not model inputs are available")

    model_output_tensors = []
    for name in model_io_cdict["outputs"]:
        try:
            node_d = built_nodes[name]
        except KeyError:
            raise KeyError(f"{name} not found in built nodes when creating outputs")

        try:
            out = node_d["out"]
        except KeyError:
            raise KeyError(
                f"out was not created for {name} when creating tensor outputs"
            )
        model_output_tensors.append(out)
    if not model_output_tensors:
        raise ValueError(f"not model outputs are available")

    # ---------------------------------------------

    # TODO: inputs may be more complex than an ordered list
    # TODO: outputs could be a list
    # TODO: right now it is assumed that the last layer defined in the config is the
    # output layer -- this may not be true. named outputs would be better.
    model = tf.keras.Model(inputs=model_input_tensors, outputs=model_output_tensors)

    # write meta.json including model hash
    if write_build_information(model_cdict, meta_cdict):
        logger.info("information json file created")

    return model
