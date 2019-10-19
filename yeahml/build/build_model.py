import tensorflow as tf

from yeahml.build.build_layers import build_hidden_block

# from yeahml.build.get_components import get_logits_and_preds
from yeahml.log.yf_logging import config_logger


# Helper to make the output "consistent"
def reset_graph_deterministic(seed=42):
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph_deterministic")
    # there is no option for deterministic behavior yet...
    # > tf issue https://github.com/tensorflow/tensorflow/issues/18096
    # os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    tf.random.set_seed(seed)
    # np.random.seed(seed)


def reset_graph(seed=42):
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph")
    # tf.reset_default_graph()
    tf.random.set_seed(seed)


def build_model(meta_cdict: dict, model_cdict: dict, log_cdict: dict, data_cdict: dict):
    # logger = logging.getLogger("build_logger")
    logger = config_logger(meta_cdict["log_dir"], log_cdict, "build")
    logger.info("-> START building graph")

    try:
        reset_graph_deterministic(meta_cdict["seed"])
    except KeyError:
        reset_graph()

    g_logger = config_logger(meta_cdict["log_dir"], log_cdict, "graph")

    # TODO: this method is a bit sloppy and I'm not sure it's needed anymore.
    # previously, the batch dimension [0], which was filled as (-1) was needed, but
    # maybe it is no longer needed with tf2. `parse_data()` is where this originally
    # created.
    if data_cdict["input_layer_dim"][0] == -1:
        input_layer = tf.keras.Input(shape=(data_cdict["input_layer_dim"][1:]))
    else:
        input_layer = tf.keras.Input(shape=(data_cdict["input_layer_dim"]))

    # TODO: this logic needs to be rethought.. Right now, despite using the functional
    # api, the process acts as a sequential api. This could change by first building all
    # all the layers and then building connecting the graph.

    # create the architecture
    hidden_layers = build_hidden_block(model_cdict, logger, g_logger)
    cur_input, cur_output = input_layer, None

    for i, layer in enumerate(hidden_layers):
        cur_output = layer(cur_input)
        cur_input = cur_output

    # TODO: need to ensure this is the API we want
    # TODO: inputs could be a list
    model = tf.keras.Model(inputs=input_layer, outputs=cur_output)

    return model
