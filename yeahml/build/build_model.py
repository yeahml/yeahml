import tensorflow as tf

from yeahml.build.build_layers import build_hidden_block
from yeahml.build.get_components import get_tf_dtype

# from yeahml.build.get_components import get_logits_and_preds
from yeahml.helper import fmt_tensor_info
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


def build_model(MCd: dict, HCd: dict):
    # logger = logging.getLogger("build_logger")
    logger = config_logger(MCd, "build")
    logger.info("-> START building graph")

    try:
        reset_graph_deterministic(MCd["seed"])
    except KeyError:
        reset_graph()

    g_logger = config_logger(MCd, "graph")

    # TODO: currently hardcoded
    input_layer = tf.keras.Input(shape=(28, 28, 1))
    # input_layer = tf.keras.Input(shape=(256,))
    print(f"jack: {MCd['input_layer_dim']}")
    # input_layer = tf.keras.Input(shape=MCd["input_layer_dim"])

    ## add layers
    ## create the architecture
    hidden_layers = build_hidden_block(MCd, HCd, logger, g_logger)

    # TODO: build model
    cur_input, cur_output = input_layer, None
    for layer in hidden_layers:
        cur_output = layer(cur_input)
        cur_input = cur_output

    # TODO: output layer
    output_layer = tf.keras.layers.Dense(10, activation="softmax")(cur_output)
    # output_layer = tf.keras.layers.Dense(1, activation="softmax")(cur_output)

    # TODO: need to ensure this is the API we want
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model
