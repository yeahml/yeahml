import tensorflow as tf
import pickle
from tqdm import tqdm
import os

from yamlflow.build.get_components import get_tf_dtype


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


def get_parse_type(parse_dict: dict):
    tfr_obj = None
    if parse_dict["tftype"] == "fixedlenfeature":
        tfr_obj = tf.FixedLenFeature([], get_tf_dtype(parse_dict["in_type"]))
    elif parse_dict["tftype"] == "fixedlensequencefeature":
        tfr_obj = tf.FixedLenSequenceFeature(
            [], get_tf_dtype(parse_dict["in_type"]), allow_missing=True
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

    if f_dict["in_type"] == "string" and f_dict["dtype"] != "string":
        image = tf.decode_raw(
            parsed_features[featureName], get_tf_dtype(f_dict["dtype"])
        )
        image = tf.reshape(image, data_in_dict["dim"])
    else:
        image = parsed_features[featureName]

    if l_dict["in_type"] == "string":
        label = tf.decode_raw(parsed_features[labelName], get_tf_dtype(l_dict["dtype"]))
        label = tf.reshape(label, data_out_dict["dim"])
    else:
        label = parsed_features[labelName]

    # TODO: One hot as needed here......
    if one_hot:
        # [-1] needed to remove the added batching
        label = tf.one_hot(label, depth=output_dim)

    # augmentation
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

