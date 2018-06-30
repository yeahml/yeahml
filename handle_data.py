import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import os


def augment_image(img_tensor):
    # TODO: this needs to be based on config
    if np.random.rand() < 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)

    # if np.random.rand() < 0.5:
    #     img_tensor = tf.image.flip_up_down(img_tensor)

    # if np.random.rand() < 0.5:
    #     img_tensor = tf.image.rot90(img_tensor)

    if np.random.rand() < 0.5:
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)

    # TODO: confirm these intervals
    if np.random.rand() < 0.5:
        img_tensor = tf.image.random_contrast(img_tensor, lower=0.0, upper=0.5)

    if np.random.rand() < 0.5:
        img_tensor = tf.image.random_hue(img_tensor, max_delta=0.3)

    if np.random.rand() < 0.5:
        img_tensor = tf.image.random_saturation(img_tensor, lower=0.0, upper=0.5)

    return img_tensor


def _parse_function(example_proto, set_type):

    labelName = str(set_type) + "/label"
    featureName = str(set_type) + "/image"
    feature = {
        featureName: tf.FixedLenFeature([], tf.string),
        labelName: tf.FixedLenFeature([], tf.int64),
    }

    # decode
    parsed_features = tf.parse_single_example(example_proto, features=feature)

    # convert image data from string to number
    image = tf.decode_raw(parsed_features[featureName], tf.float32)
    # TODO: these values should be acquired from the yaml
    image = tf.reshape(image, [150, 150, 3])
    label = tf.cast(parsed_features[labelName], tf.int64)

    # TODO: will need to figure out how to aug_opts information here.
    # NOTE: Augmentation! this may not be the best place to do this.
    if set_type != "test":
        # TODO: this needs to be based on config
        image = augment_image(image)

    # TODO: this needs to be based on config
    image = tf.image.per_image_standardization(image)

    return image, label


def return_batched_iter(set_type, MCd, filenames_ph):

    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(
        lambda x: _parse_function(x, set_type)
    )  # Parse the record into tensors.
    if set_type != "test":
        dataset = dataset.shuffle(buffer_size=MCd["shuffle_buffer"])
    # dataset = dataset.shuffle(buffer_size=1)
    # prefetch is used to ensure one batch is always ready
    # TODO: this prefetch should have some logic based on the
    # system environment, batchsize, and data size
    dataset = dataset.batch(MCd["batch_size"]).prefetch(1)
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    return iterator

