import tensorflow as tf
import math
from tqdm import tqdm
import os
import sys
from typing import Any

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging
from yeahml.build.load_params_onto_layer import init_params_from_file  # load params
from yeahml.build.get_components import get_optimizer


def train_model(model, MCd: dict, HCd: dict) -> dict:
    return_dict = {}

    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    # get model
    # get optimizer
    optimizer = get_optimizer(MCd)
    # get loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # get dataset
    tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
    train_ds = return_batched_iter("train", MCd, tfr_f_path)

    # TODO: train loop

    for e in range(MCd["epochs"]):
        logger.debug("-> START iterating training dataset")
        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):

            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
                print(
                    "Training loss (for single batch) at step %s: %s"
                    % (step, float(loss_value))
                )

    logger.info("[END] creating train_dict")
