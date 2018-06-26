import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import os

TFR_DIR = "./example/cats_v_dogs_01/data/record_holder/150"


def augment_image(img_tensor):
    # TODO: this needs to be based on config
    if np.random.rand() < 0.5:
        img_tensor = tf.image.flip_left_right(img_tensor)

    # if np.random.rand() < 0.5:
    # img_tensor = tf.image.flip_up_down(img_tensor)

    # if np.random.rand() < 0.5:
    # img_tensor = tf.image.rot90(img_tensor)

    if np.random.rand() < 0.5:
        img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.2)

    return img_tensor


def _parse_function(example_proto):
    global GLOBAL_SET_TYPE
    labelName = str(GLOBAL_SET_TYPE) + "/label"
    featureName = str(GLOBAL_SET_TYPE) + "/image"
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

    # [do any preprocessing here]
    # TODO: this needs to be based on config
    image = tf.image.per_image_standardization(image)

    # TODO: will need to figure out how to aug_opts information here.
    # NOTE: Augmentation! this may not be the best place to do this.
    if GLOBAL_SET_TYPE != "test":
        # TODO: this needs to be based on config
        image = augment_image(image)

    return image, label


def return_batched_iter(setType, MCd, sess):
    global GLOBAL_SET_TYPE
    global TFR_DIR
    GLOBAL_SET_TYPE = setType

    filenames_ph = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    if GLOBAL_SET_TYPE != "test":
        dataset = dataset.shuffle(buffer_size=MCd["shuffle_buffer"])
    # dataset = dataset.shuffle(buffer_size=1)
    # prefetch is used to ensure one batch is always ready
    # TODO: this prefetch should have some logic based on the
    # system environment, batchsize, and data size
    dataset = dataset.batch(MCd["batch_size"]).prefetch(1)
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    tfrecords_file_name = str(GLOBAL_SET_TYPE) + ".tfrecords"
    tfrecord_file_path = os.path.join(TFR_DIR, tfrecords_file_name)

    # initialize
    sess.run(iterator.initializer, feed_dict={filenames_ph: [tfrecord_file_path]})

    return iterator
