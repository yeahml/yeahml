import os
import pickle
import random

import tensorflow as tf
from tqdm import tqdm

from yeahml.build.get_components import get_tf_dtype


def augment_image(img_tensor, gt_tensor, aug_opts: dict) -> tuple:
    raise NotImplementedError


def get_parse_type(parse_dict: dict):
    tfr_obj = None
    tf_parse_type = parse_dict["tftype"].lower()
    if tf_parse_type == "fixedlenfeature":
        tfr_obj = tf.io.FixedLenFeature([], get_tf_dtype(parse_dict["in_type"]))
    elif tf_parse_type == "fixedlensequencefeature":
        tfr_obj = tf.io.FixedLenSequenceFeature(
            [], get_tf_dtype(parse_dict["in_type"]), allow_missing=True
        )
    else:
        raise ValueError(
            "tf_parse_type: {tf_parse_type} -- is not supported or defined"
        )
    return tfr_obj


def _parse_function(
    example_proto,
    set_type: str,
    standardize_img: bool,
    aug_opts: dict,
    parse_shape: list,
    one_hot: bool,
    output_dim: int,
    data_in_dict: dict,
    data_out_dict: dict,
    tfr_parse_dict: dict,
):

    # TODO: this logic is sloppy.. and needs to be better organized

    f_dict = tfr_parse_dict["feature"]
    featureName = f_dict["name"]

    l_dict = tfr_parse_dict["label"]
    labelName = l_dict["name"]

    feature_dict = {
        featureName: get_parse_type(f_dict),
        labelName: get_parse_type(l_dict),
    }

    # decode
    parsed_features = tf.io.parse_single_example(example_proto, features=feature_dict)

    ## Feature

    # decode string
    if f_dict["in_type"] == "string" and f_dict["dtype"] != "string":
        image = tf.io.decode_raw(
            parsed_features[featureName], get_tf_dtype(f_dict["dtype"])
        )
        image = tf.dtypes.cast(image, dtype=tf.float32)
    else:
        image = parsed_features[featureName]

    # if a reshape is present in the config for the feature, reshape the data
    try:
        if data_in_dict["reshape_to"]:
            image = tf.reshape(image, parse_shape)
    except KeyError:
        image = tf.reshape(image, parse_shape)

    # try:
    #     if data_in_dict["cast_to"]:

    # except KeyError:
    #     image = tf.reshape(image, parse_shape)

    ## Label

    # decode string
    if l_dict["in_type"] == "string":
        label = tf.decode_raw(parsed_features[labelName], get_tf_dtype(l_dict["dtype"]))
        # if a reshape is present in the config for the label, reshape the data
    else:
        label = parsed_features[labelName]

    if one_hot:
        # [-1] needed to remove the added batching
        label = tf.one_hot(label, depth=output_dim)

    # handle shape
    try:
        if data_out_dict["reshape_to"]:
            label = tf.reshape(label, data_out_dict["reshape_to"])
    except KeyError:
        label = tf.reshape(label, data_out_dict["dim"])

    # augmentation
    if aug_opts:
        if set_type == "train":
            image, label = augment_image(image, label, aug_opts)
        elif set_type == "val":
            try:
                if aug_opts["aug_val"]:
                    image, label = augment_image(image, label, aug_opts)
            except KeyError:
                pass
        else:  # test
            pass

    if standardize_img:
        image = tf.image.per_image_standardization(image)

    # return image, label, inst_id
    return image, label


def return_batched_iter(set_type: str, MCd: dict, tfr_f_path):

    try:
        aug_opts = MCd["augmentation"]
    except KeyError:
        aug_opts = None

    try:
        standardize_img = MCd["image_standardize"]
    except KeyError:
        standardize_img = False

    # TODO: revamp this. This calculation should be performed in the parse fn
    # the config file will also need to be adjusted
    if MCd["reshape_in_to"]:
        parse_shape = MCd["reshape_in_to"]
        if MCd["reshape_in_to"][0] != -1:
            parse_shape = MCd["reshape_in_to"]
        else:  # e.g. [-1, 28, 28, 1]
            parse_shape = MCd["reshape_in_to"][1:]
    else:
        if MCd["in_dim"][0]:
            parse_shape = MCd["in_dim"]
        else:  # e.g. [None, 28, 28, 1]
            parse_shape = MCd["in_dim"][1:]
    # parse_shape = list(parse_shape)

    # TODO: implement directory or files logic
    # tf.data.Dataset.list_files
    dataset = tf.data.TFRecordDataset(tfr_f_path)
    dataset = dataset.map(
        lambda x: _parse_function(
            x,
            set_type,
            standardize_img,
            aug_opts,
            parse_shape,
            MCd["label_one_hot"],
            MCd["output_dim"][-1],  # used for one_hot encoding
            MCd["data_in_dict"],
            MCd["data_out_dict"],
            MCd["TFR_parse"],
        )
    )  # Parse the record into tensors.
    if set_type == "train":
        dataset = dataset.shuffle(buffer_size=MCd["shuffle_buffer"])
    # dataset = dataset.shuffle(buffer_size=1)
    # prefetch is used to ensure one batch is always ready
    # TODO: this prefetch should have some logic based on the
    # system environment, batchsize, and data size
    dataset = dataset.batch(MCd["batch_size"]).prefetch(1)
    dataset = dataset.repeat(1)

    # iterator = dataset.make_initializable_iterator()

    return dataset
