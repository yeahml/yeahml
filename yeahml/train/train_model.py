import tensorflow as tf
import math
from tqdm import tqdm
import os
import sys
from typing import Any

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.build.load_params_onto_layer import init_params_from_file  # load params


def train_model(g, MCd: dict, HCd: dict) -> dict:
    return_dict = {}

    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    # TODO: train

    raise NotImplementedError

    logger.info("[END] creating train_dict")
