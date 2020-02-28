from typing import Any, Dict

import tensorflow as tf
from pathlib import Path

from yeahml.build.build_layers import build_hidden_block
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


def build_model(config_dict: Dict[str, Dict[str, Any]]) -> Any:

    # unpack configuration
    model_cdict: Dict[str, Any] = config_dict["model"]
    meta_cdict: Dict[str, Any] = config_dict["meta"]
    log_cdict: Dict[str, Any] = config_dict["logging"]
    data_cdict: Dict[str, Any] = config_dict["data"]

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

    # loop through the graph to find all inputs required from the data

    # TODO: this method is a bit sloppy and I'm not sure it's needed anymore.
    # previously, the batch dimension [0], which was filled as (-1) was needed, but
    # maybe it is no longer needed with tf2. `parse_data()` is where this originally
    # created.
    # TODO: remove this from the data_cdict
    # if data_cdict["input_layer_dim"][0] == -1:
    #     input_layer = tf.keras.Input(shape=(data_cdict["input_layer_dim"][1:]))
    # else:
    #     input_layer = tf.keras.Input(shape=(data_cdict["input_layer_dim"]))

    # TODO: this logic needs to be rethought.. Right now, despite using the functional
    # api, the process acts as a sequential api. This could change by first building all
    # all the layers and then building connecting the graph.

    # create the architecture
    print(model_cdict)
    hidden_layers = build_hidden_block(model_cdict, logger, g_logger)
    print(hidden_layers)
    sys.exit()
    cur_input, cur_output = input_layer, None

    # TODO: we could check for graph things here - heads/ends, cycles, etc.
    # TODO: Not sure if BF or DF would be better here when building the graph

    graph_dict: Dict[str, Any] = {}
    for layer_name, layer_dict in hidden_layers.items():
        graph_dict[layer_name] = {}
        layer_fn = layer_dict["layer_fn"]
        input_str = layer_dict["input_str"]

        if input_str == "data_input":
            layer_input = input_layer
        else:
            try:
                # obtain previous layer output

                # NOTE: by accessing the ["layer_output"] here. we are preventing a layer from
                # specifying itself as the input. since the ["layer_output"] has not been defined yet.
                # the error message could be improved here.
                layer_input = graph_dict[input_str]["layer_output"]
            except KeyError:
                if input_str in hidden_layers.keys():
                    raise ValueError(
                        f"layer {input_str} has not been created yet (so far: {graph_dict.keys()}). please move {input_str} up in the config file"
                    )
                else:
                    raise ValueError(
                        f"layer {input_str} is not defined in the config file. The defined layers are ({hidden_layers.keys()}). please check name spelling or define {input_str}"
                    )

        graph_dict[layer_name]["layer_input"] = layer_input

        cur_layer_out = layer_fn(layer_input)
        graph_dict[layer_name]["layer_output"] = cur_layer_out
        # cur_output = layer_fn(cur_input)
        # cur_input = cur_output

    # TODO: Analyze graph_dict to see if there are any "dead_ends"

    # TODO: depending on the implementation, we may need to create multiple models
    # based on the `graph_dict` components. This is also likely where we should log
    # the graph information/structure

    # TODO: need to ensure this is the API we want
    # TODO: inputs could be a list
    # TODO: outputs could be a list
    # TODO: right now it is assumed that the last layer defined in the config is the
    # output layer -- this may not be true. named outputs would be nice.
    model = tf.keras.Model(inputs=input_layer, outputs=cur_layer_out)

    # write meta.json including model hash
    if write_build_information(model_cdict, meta_cdict):
        logger.info("information json file created")

    return model
