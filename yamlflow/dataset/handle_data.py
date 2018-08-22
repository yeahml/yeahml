import tensorflow as tf
import pickle
from tqdm import tqdm
import os
import sys
import random

from yamlflow.build.get_components import get_tf_dtype


def augment_image(img_tensor, gt_tensor, aug_opts: dict) -> tuple:

    # TODO: There is an issue here in which augmentation is working "as expected" but yet does
    # not work correctly in that the seed value does not change and thus the images always undergo
    # the same transformation -- that is, the image and mask are both augmented correctly, but in each
    # iteration the image and mask are augmented the same way, meaning, they are only augmented "once"
    # throughout all iterations

    # TODO: temp > AUG_LABEL is used to determine if the label should also
    # be augmented to match the image augmentation

    try:
        AUG_LABEL = aug_opts["label"]
    except KeyError:
        AUG_LABEL = False

    if AUG_LABEL:
        gt_tensor = tf.expand_dims(gt_tensor, -1)

    try:
        if aug_opts["h_flip"]:
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_flip_left_right(img_tensor, seed=seed)
            if AUG_LABEL:
                # print("HERE+{}".format(seed))
                gt_tensor = tf.image.random_flip_left_right(gt_tensor, seed=seed)
    except KeyError:
        pass

    try:
        if aug_opts["v_flip"]:
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_flip_up_down(img_tensor, seed=seed)
            if AUG_LABEL:
                gt_tensor = tf.image.random_flip_up_down(gt_tensor, seed=seed)
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
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_brightness(
                img_tensor, max_delta=brt_max, seed=seed
            )
            if AUG_LABEL:
                gt_tensor = tf.image.random_brightness(
                    gt_tensor, max_delta=brt_max, seed=seed
                )
    except KeyError:
        pass

    # TODO: confirm these intervals
    try:
        contrast_val = aug_opts["contrast"]
        # if np.random.rand() < 0.5:
        if contrast_val > 0:
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_contrast(
                img_tensor, lower=0.0, upper=contrast_val, seed=seed
            )
            if AUG_LABEL:
                gt_tensor = tf.image.random_contrast(
                    gt_tensor, lower=0.0, upper=contrast_val, seed=seed
                )
    except KeyError:
        pass

    try:
        hue_max = aug_opts["hue"]
        # if np.random.rand() < 0.5:
        if hue_max > 0:
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_hue(img_tensor, max_delta=hue_max, seed=seed)
            if AUG_LABEL:
                gt_tensor = tf.image.random_hue(gt_tensor, max_delta=hue_max, seed=seed)
    except KeyError:
        pass

    try:
        sat_val = aug_opts["saturation"]
        # if np.random.rand() < 0.5:
        if sat_val > 0:
            seed = random.randint(1, 100)
            img_tensor = tf.image.random_saturation(
                img_tensor, lower=0.0, upper=sat_val, seed=seed
            )
            if AUG_LABEL:
                gt_tensor = tf.image.random_saturation(
                    gt_tensor, lower=0.0, upper=sat_val, seed=seed
                )
    except KeyError:
        pass

    if AUG_LABEL:
        gt_tensor = tf.squeeze(gt_tensor)

    print("AUG_LABEL = {}".format(AUG_LABEL))

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

