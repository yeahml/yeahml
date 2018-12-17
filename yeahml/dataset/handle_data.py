import tensorflow as tf
import pickle
from tqdm import tqdm
import os
import sys
import random

from yeahml.build.get_components import get_tf_dtype


def augment_image(img_tensor, gt_tensor, aug_opts: dict) -> tuple:

    # TODO: temp > AUG_LABEL is used to determine if the label should also
    # be augmented to match the image augmentation

    # NOTE: each 'block' uses a tf.where() clause controlled by a tf.random_uniform
    # value to ensure the same augmentation is applied to the image and label

    try:
        AUG_LABEL = aug_opts["label"]
    except KeyError:
        AUG_LABEL = False

    if AUG_LABEL:
        # expand label dims to a 3D tensor > TODO: there [sh|co]uld be a check here
        gt_tensor = tf.expand_dims(gt_tensor, -1)

    try:
        if aug_opts["h_flip"]:
            # seed = random.randint(1, 100)
            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.where(
                tf.less(tf.constant(3), cond_num),
                tf.image.flip_left_right(img_tensor),
                img_tensor,
            )

            if AUG_LABEL:
                gt_tensor = tf.where(
                    tf.less(tf.constant(3), cond_num),
                    tf.image.flip_left_right(gt_tensor),
                    gt_tensor,
                )

    except KeyError:
        pass

    try:
        if aug_opts["v_flip"]:
            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.where(
                tf.less(tf.constant(3), cond_num),
                tf.image.flip_up_down(img_tensor),
                img_tensor,
            )

            if AUG_LABEL:
                gt_tensor = tf.where(
                    tf.less(tf.constant(3), cond_num),
                    tf.image.flip_up_down(gt_tensor),
                    gt_tensor,
                )
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
        bright_max = aug_opts["max_brightness"]
        # if np.random.rand() < 0.5:
        if bright_max > 0:
            # TODO: include initial check to ensure value is below 1
            bright_val = tf.random_uniform(
                [], minval=0, maxval=bright_max, dtype=tf.float32
            )

            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.where(
                tf.less(tf.constant(3), cond_num),
                tf.image.adjust_brightness(img_tensor, delta=bright_val),
                img_tensor,
            )

            if AUG_LABEL:
                gt_tensor = tf.where(
                    tf.less(tf.constant(3), cond_num),
                    tf.image.adjust_brightness(gt_tensor, delta=bright_val),
                    gt_tensor,
                )

    except KeyError:
        pass

    # TODO: confirm these intervals
    try:
        contrast_max = aug_opts["contrast"]
        # if np.random.rand() < 0.5:
        if contrast_max > 0:
            contrast_val = tf.random_uniform(
                [], minval=0, maxval=contrast_max, dtype=tf.float32
            )
            # TODO: sanity check the contrast value (maybe in the config file)
            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.where(
                tf.less(tf.constant(3), cond_num),
                tf.image.adjust_contrast(img_tensor, contrast_factor=contrast_val),
                img_tensor,
            )

            if AUG_LABEL:
                gt_tensor = tf.where(
                    tf.less(tf.constant(3), cond_num),
                    tf.image.adjust_contrast(gt_tensor, contrast_factor=contrast_val),
                    gt_tensor,
                )

    except KeyError:
        pass

    try:
        hue_max = aug_opts["hue"]
        # if np.random.rand() < 0.5:
        if hue_max > 0:
            hue_val = tf.random_uniform([], minval=0, maxval=hue_max, dtype=tf.float32)
            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.image.adjust_hue(img_tensor, delta=hue_val)
            if AUG_LABEL:
                gt_tensor = tf.image.adjust_hue(gt_tensor, delta=hue_val)
    except KeyError:
        pass

    try:
        sat_max = aug_opts["saturation"]
        # if np.random.rand() < 0.5:
        if sat_max > 0:
            sat_val = tf.random_uniform([], minval=0, maxval=sat_max, dtype=tf.float32)
            cond_num = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
            img_tensor = tf.image.adjust_saturation(
                img_tensor, saturation_factor=sat_val
            )
            if AUG_LABEL:
                gt_tensor = tf.image.adjust_saturation(
                    gt_tensor, saturation_factor=sat_val
                )
    except KeyError:
        pass

    if AUG_LABEL:
        # squeeze back into expected form (remove the expanded dim)
        gt_tensor = tf.squeeze(gt_tensor)

    return (img_tensor, gt_tensor)


def get_parse_type(parse_dict: dict):
    tfr_obj = None
    tf_parse_type = parse_dict["tftype"].lower()
    if tf_parse_type == "fixedlenfeature":
        tfr_obj = tf.FixedLenFeature([], get_tf_dtype(parse_dict["in_type"]))
    elif tf_parse_type == "fixedlensequencefeature":
        tfr_obj = tf.FixedLenSequenceFeature(
            [], get_tf_dtype(parse_dict["in_type"]), allow_missing=True
        )
    else:
        sys.exit(
            "tf_parse_type: {} -- is not supported or defined.".format(tf_parse_type)
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
    parsed_features = tf.parse_single_example(example_proto, features=feature_dict)

    ## Feature

    # decode string
    if f_dict["in_type"] == "string" and f_dict["dtype"] != "string":
        image = tf.decode_raw(
            parsed_features[featureName], get_tf_dtype(f_dict["dtype"])
        )
    else:
        image = parsed_features[featureName]

    # if a reshape is present in the config for the feature, reshape the data
    try:
        if data_in_dict["reshape_to"]:
            image = tf.reshape(image, parse_shape)
    except KeyError:
        image = tf.reshape(image, parse_shape)

    ## Label

    # decode string
    if l_dict["in_type"] == "string":
        label = tf.decode_raw(parsed_features[labelName], get_tf_dtype(l_dict["dtype"]))
        # if a reshape is present in the config for the label, reshape the data
    else:
        label = parsed_features[labelName]

    # TODO: One hot as needed here......
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


def return_batched_iter(set_type: str, MCd: dict, filenames_ph):

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

    dataset = tf.data.TFRecordDataset(filenames_ph)
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

    iterator = dataset.make_initializable_iterator()

    return iterator

