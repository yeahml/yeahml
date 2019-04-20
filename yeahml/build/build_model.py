import tensorflow as tf
import sys

from yeahml.log.yf_logging import config_logger
from yeahml.build.build_layers import build_hidden_block
from yeahml.build.get_components import get_tf_dtype
from yeahml.build.get_components import get_optimizer
from yeahml.build.get_components import get_logits_and_preds
from yeahml.build.helper import create_metrics_ops
from yeahml.helper import fmt_tensor_info


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

    ## make input layer(s?)
    MODEL = None  # sequential model?

    ## add layers
    ## create the architecture
    MODEL = build_hidden_block(MODEL, MCd, HCd, logger, g_logger)

    return MODEL
