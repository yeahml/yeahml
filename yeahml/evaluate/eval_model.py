import tensorflow as tf
import os
import numpy as np
from typing import Any

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging


def eval_model(model: Any, MCd: dict, weights_path: str = None) -> dict:

    logger = config_logger(MCd, "eval")
    logger.info("-> START evaluating model")

    # load best weights
    # TODO: load specific weights according to a param
    if weights_path:
        specified_path = weights_path
    else:
        specified_path = MCd["save_weights_path"]

    model.load_weights(specified_path)
    logger.info("params loaded from {}".format(specified_path))

    logger.info("[END] creating train_dict")

    eval_dict = {}
    return eval_dict

    raise NotImplementedError

