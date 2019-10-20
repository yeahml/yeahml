# TODO


def format_data_config(raw_config):
    formatted_dict = {}
    ## inputs
    # TODO: this is sloppy...
    # copy is used to prevent overwriting underlying data
    formatted_dict["input_layer_dim"] = None
    formatted_dict["in_dim"] = raw_config["in"]["dim"].copy()
    if formatted_dict["in_dim"][0]:  # as oppposed to [None, x, y, z]
        formatted_dict["in_dim"].insert(0, None)  # add batching
    formatted_dict["in_dtype"] = raw_config["in"]["dtype"]
    try:
        formatted_dict["reshape_in_to"] = raw_config["in"]["reshape_to"]
        if formatted_dict["reshape_in_to"][0] != -1:  # as oppposed to [None, x, y, z]
            formatted_dict["reshape_in_to"].insert(0, -1)  # -1
    except KeyError:
        # None in this case is representative of not reshaping
        formatted_dict["reshape_in_to"] = None
    if formatted_dict["reshape_in_to"]:
        formatted_dict["input_layer_dim"] = raw_config["in"]["reshape_to"]
    else:
        formatted_dict["input_layer_dim"] = raw_config["in"]["dim"].copy()

    # copy is used to prevent overwriting underlying data
    formatted_dict["output_dim"] = raw_config["label"]["dim"].copy()
    if formatted_dict["output_dim"][0]:  # as oppposed to [None, x, y, z]
        formatted_dict["output_dim"].insert(0, None)  # add batching
    formatted_dict["label_dtype"] = raw_config["label"]["dtype"]

    try:
        temp_one_hot = raw_config["label"]["one_hot"]
        if temp_one_hot != False and temp_one_hot != True:
            raise ValueError(
                f"Error > Exiting: data:label:one_hot {temp_one_hot} unsupported. Please use True or False"
            )
        formatted_dict["label_one_hot"] = temp_one_hot
    except KeyError:
        # None in this case is representative of not using one hot encoding
        formatted_dict["label_one_hot"] = False

    # currently required
    formatted_dict["TFR_dir"] = raw_config["TFR"]["dir"]
    formatted_dict["TFR_train"] = raw_config["TFR"]["train"]
    formatted_dict["TFR_test"] = raw_config["TFR"]["test"]
    formatted_dict["TFR_val"] = raw_config["TFR"]["validation"]

    # TODO: this is a first draft for this type of organization and will
    # will likely be changed
    formatted_dict["data_in_dict"] = raw_config["in"]
    formatted_dict["data_out_dict"] = raw_config["label"]
    formatted_dict["TFR_parse"] = raw_config["TFR_parse"]

    try:
        formatted_dict["augmentation"] = raw_config["image"]["augmentation"]
    except KeyError:
        pass

    try:
        formatted_dict["image_standardize"] = raw_config["image"]["standardize"]
    except KeyError:
        pass

    return formatted_dict
