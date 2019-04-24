import tensorflow as tf
import math
from tqdm import tqdm
import os
import sys
from typing import Any

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.build.load_params_onto_layer import init_params_from_file  # load params


def train_model(model, MCd: dict, HCd: dict) -> dict:
    return_dict = {}

    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    # TODO: train
    tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    tr_iter = return_batched_iter("train", MCd, tfr_f_path)

    logger.debug("-> START iterating training dataset")
    for data, target in tr_iter:
        print(data.shape)

    logger.info("[END] creating train_dict")
