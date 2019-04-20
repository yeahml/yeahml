import tensorflow as tf
import os
import numpy as np

from yeahml.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yeahml.log.yf_logging import config_logger  # custom logging


def eval_model(Model, MCd):
    raise NotImplementedError

