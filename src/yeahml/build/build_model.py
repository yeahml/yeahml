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
        if user_values:
            user_val = user_values[i]
        else:
            # TODO: this logic will need to be rethought, since it's possible
            # that a user argument could be `None` (I think?). I think a
            # potential solution would be to use a custom class sentinel value
            user_val = None

        # NOTE: `if user_val`: does not work because `user_val` could be False
        # TODO: this function will need to be tested
        if user_val is not None:
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


###############################################


def _is_tensor_or_list_of_tensors(obj):
    if isinstance(obj, tf.Tensor):
        return True
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, tf.Tensor):
                pass
            else:
                return False
        return True
    else:
        raise ValueError(f"prev_out is neither a tensor or list. is type: {type(obj)}")


def _build_branch(branch_seq, built_nodes):
    prev_out = None
    for sub_seq in branch_seq:
        tensors = []
        if isinstance(sub_seq, str):
            sub_seq = [sub_seq]
        for node_name in sub_seq:
            # get node
            try:
                cur_built_node = built_nodes[node_name]
            except KeyError:
                # TODO: this message will have to be expanded for huge graphs
                raise KeyError(
                    f"node ({node_name}) not found in built nodes {built_nodes.keys()}"
                )

            # get output of node
            if isinstance(cur_built_node["out"], tf.Tensor):
                # output already exists
                out = cur_built_node["out"]
            else:
                # output does not exist (func hasn't been called), get prev_out
                # to use as input
                # TODO: here
                if _is_tensor_or_list_of_tensors(
                    prev_out
                ):  # isinstance(prev_out, tf.Tensor):
                    try:
                        cur_func = cur_built_node["func"]
                    except KeyError:
                        raise KeyError(
                            f"node ({node_name}) function has not been built yet"
                        )

                    out = cur_func(prev_out)
                    cur_built_node["out"] = out
                else:
                    raise ValueError(
                        f"node ({node_name}) does not have an output yet and there is not previous out to use as an input"
                    )
                    # try:
                    #     prev_out = cur_built_node["out"]
                    # except KeyError:
                    #     raise KeyError(
                    #         f"node ({cur_name_in_seq}) does has not been created yet"
                    #     )
            tensors.append(out)
        if len(tensors) == 1:
            tensors = tensors[0]
        prev_out = tensors


def _build_sequence(seq_info, built_nodes):

    seq_to_build = []
    for obj in seq_info:
        tuple_outputs = []
        if isinstance(obj, tuple):
            # [(num_branches, [[n_a, n_b, ...],[n_c, n_d, ...]], out_name),
            # out_name, next_Nodes...]
            name = obj[1]
            branch_tuple = obj[0]
            num_branches = branch_tuple[0]
            branch_seqs = branch_tuple[1]
            for branch_seq in branch_seqs:
                _build_branch(branch_seq, built_nodes)
                tuple_outputs.append(branch_seq[-1])
            seq_to_build.append(tuple_outputs)
            # NOTE: this is a pretty weak assertion
            assert (
                len(tuple_outputs) == num_branches
            ), f"num_branches{num_branches} != {len(tuple_outputs)}"
            # each branch should now be built
            # now, jump out and build the next node/sequence
        else:
            seq_to_build.append(obj)

    _build_branch(seq_to_build, built_nodes)


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

    # for end_name, subgraph in subgraphs_cdict.items():
    #     print(f"{end_name}: {subgraph}")

    # connect the subgraphs
    for end_name, seq_dict in subgraphs_cdict.items():
        # print(f"{name}: {seq_dict}")
        seq_info = seq_dict["sequence"]
        _build_sequence(seq_info, built_nodes)

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
