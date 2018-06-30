import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import os


def augment_image(img_tensor, aug_opts: dict):

    try:
        # h_flip = aug_opts["h_flip"]
        # if np.random.rand() < h_flip:
        if aug_opts["h_flip"]:
            img_tensor = tf.image.random_flip_left_right(img_tensor)
    except KeyError:
        pass

    try:
        # v_flip = aug_opts["v_flip"]
        # if np.random.rand() < v_flip:
        if aug_opts["v_flip"]:
            img_tensor = tf.image.random_flip_up_down(img_tensor)
    except KeyError:
        pass

    # try:
    #     v_flip = aug_opts["rotate"]
    #     if np.random.rand() < v_flip:
    #         img_tensor = tf.image.rot90(img_tensor)
    # except KeyError:
    #     pass

    ##################################### image manipulation
    # TODO: confirm intervals
    # TODO: handle min/max value
    # TODO: current default is set to 0.5 > half, if specified, will be altered
    try:
        brt_max = aug_opts["max_brightness"]
        # if np.random.rand() < 0.5:
        if brt_max > 0:
            img_tensor = tf.image.random_brightness(img_tensor, max_delta=brt_max)
    except KeyError:
        pass

    # TODO: confirm these intervals
    try:
        contrast_val = aug_opts["contrast"]
        # if np.random.rand() < 0.5:
        if contrast_val > 0:
            img_tensor = tf.image.random_contrast(
                img_tensor, lower=0.0, upper=contrast_val
            )
    except KeyError:
        pass

    try:
        hue_max = aug_opts["hue"]
        # if np.random.rand() < 0.5:
        if hue_max > 0:
            img_tensor = tf.image.random_hue(img_tensor, max_delta=hue_max)
    except KeyError:
        pass

    try:
        sat_val = aug_opts["saturation"]
        # if np.random.rand() < 0.5:
        if sat_val > 0:
            img_tensor = tf.image.random_saturation(
                img_tensor, lower=0.0, upper=sat_val
            )
    except KeyError:
        pass

    return img_tensor


def _parse_function(
    example_proto, set_type: str, standardize_img: bool, aug_opts: dict
):

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

    # Augmentation
    if aug_opts:
        if set_type == "train":
            image = augment_image(image, aug_opts)
        elif set_type == "val":
            try:
                if aug_opts["aug_val"]:
                    image = augment_image(image, aug_opts)
            except KeyError:
                pass
        else:  # test
            pass

    if standardize_img:
        image = tf.image.per_image_standardization(image)

    return image, label


def return_batched_iter(set_type: str, MCd: dict, filenames_ph):

    try:
        aug_opts = MCd["augmentation"]
    except KeyError:
        aug_opts = None

    try:
        standardize_img = MCd["image_standardize"]
    except KeyError:
        standardize_img = False

    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(
        lambda x: _parse_function(x, set_type, standardize_img, aug_opts)
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

