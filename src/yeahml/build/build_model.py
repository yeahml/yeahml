from pathlib import Path
from typing import Any, Dict

import tensorflow as tf

from yeahml.build.layers.config import NOTPRESENT
from yeahml.config.graph_analysis.build_graph_dict import get_node_config_by_name
from yeahml.information.write_info import write_build_information

# from yeahml.build.get_components import get_logits_and_preds
from yeahml.log.yf_logging import config_logger

# def get_lr_schedule():
#     # TODO: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/optimizers/schedules/ExponentialDecay
#     raise NotImplementedError


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
        return False


def _is_valid_output(obj):
    if _is_tensor_or_list_of_tensors(obj):
        return True
    else:
        return False


def build_model(config_dict: Dict[str, Dict[str, Any]]) -> Any:

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    # data_cdict: Dict[str, Any] = config_dict["data"]
    graph_dict: Dict[str, Any] = config_dict["graph_dict"]
    graph_dependencies: Dict[str, Any] = config_dict["graph_dependencies"]

    model_io_cdict: Dict[str, Any] = config_dict["model_io"]

    full_exp_path = (
        Path(meta_cdict["yeahml_dir"])
        .joinpath(meta_cdict["data_name"])
        .joinpath(meta_cdict["experiment_name"])
        .joinpath(model_cdict["name"])
    )
    logger = config_logger(full_exp_path, log_cdict, "build")
    logger.info("-> START building graph")

    try:
        reset_graph_deterministic(meta_cdict["seed"])
    except KeyError:
        reset_graph()

    # g_logger = config_logger(full_exp_path, log_cdict, "graph")

    # configure/build all layers and save in lookup table
    built_nodes = {}
    # {"<layer_name>": {"func": <layer_func>, "out": <output_of_layer>}}
    for name, node in graph_dict.items():

        node_config = get_node_config_by_name(node.name, config_dict)
        if not node_config:
            raise ValueError(f"layer {name} can't be found in {node.config_location}")
        blueprint = node_config["object_dict"]

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

    for group_of_nodes in graph_dependencies:
        list_of_nodes = list(group_of_nodes)
        for cur_node in list_of_nodes:
            if cur_node:
                if not graph_dict[cur_node].label:
                    if not _is_valid_output(built_nodes[cur_node]["out"]):
                        # create the output (it doesn't exist yet)
                        in_names = graph_dict[cur_node].in_name
                        prev_outputs = []
                        for in_name in in_names:
                            prev_out = built_nodes[in_name]["out"]
                            prev_outputs.append(prev_out)

                        # if only one previous output is present, remove list
                        if len(prev_outputs) == 1:
                            prev_outputs = prev_outputs[0]

                        # connect
                        cur_out = built_nodes[cur_node]["func"](prev_outputs)
                        built_nodes[cur_node]["out"] = cur_out
                    else:
                        pass

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
    model = tf.keras.Model(
        inputs=model_input_tensors,
        outputs=model_output_tensors,
        name=model_cdict["name"],
    )

    # write meta.json including model hash
    if write_build_information(model_cdict, meta_cdict):
        logger.info("information json file created")

    return model
