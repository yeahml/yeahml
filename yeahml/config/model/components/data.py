# TODO


def parse_data(MC: dict) -> dict:
    MCd = {}
    ## inputs
    # TODO: this is sloppy...
    # copy is used to prevent overwriting underlying data
    MCd["input_layer_dim"] = None
    MCd["in_dim"] = MC["data"]["in"]["dim"].copy()
    if MCd["in_dim"][0]:  # as oppposed to [None, x, y, z]
        MCd["in_dim"].insert(0, None)  # add batching
    MCd["in_dtype"] = MC["data"]["in"]["dtype"]
    try:
        MCd["reshape_in_to"] = MC["data"]["in"]["reshape_to"]
        if MCd["reshape_in_to"][0] != -1:  # as oppposed to [None, x, y, z]
            MCd["reshape_in_to"].insert(0, -1)  # -1
    except KeyError:
        # None in this case is representative of not reshaping
        MCd["reshape_in_to"] = None
    if MCd["reshape_in_to"]:
        MCd["input_layer_dim"] = MC["data"]["in"]["reshape_to"]
    else:
        MCd["input_layer_dim"] = MC["data"]["in"]["dim"].copy()

    # copy is used to prevent overwriting underlying data
    MCd["output_dim"] = MC["data"]["label"]["dim"].copy()
    if MCd["output_dim"][0]:  # as oppposed to [None, x, y, z]
        MCd["output_dim"].insert(0, None)  # add batching
    MCd["label_dtype"] = MC["data"]["label"]["dtype"]

    try:
        temp_one_hot = MC["data"]["label"]["one_hot"]
        if temp_one_hot != False and temp_one_hot != True:
            raise ValueError(
                f"Error > Exiting: data:label:one_hot {temp_one_hot} unsupported. Please use True or False"
            )
        MCd["label_one_hot"] = temp_one_hot
    except KeyError:
        # None in this case is representative of not using one hot encoding
        MCd["label_one_hot"] = False

    # currently required
    MCd["TFR_dir"] = MC["data"]["TFR"]["dir"]
    MCd["TFR_train"] = MC["data"]["TFR"]["train"]
    MCd["TFR_test"] = MC["data"]["TFR"]["test"]
    MCd["TFR_val"] = MC["data"]["TFR"]["validation"]

    # TODO: this is a first draft for this type of organization and will
    # will likely be changed
    MCd["data_in_dict"] = MC["data"]["in"]
    MCd["data_out_dict"] = MC["data"]["label"]
    MCd["TFR_parse"] = MC["data"]["TFR_parse"]

    try:
        MCd["augmentation"] = MC["data"]["image"]["augmentation"]
    except KeyError:
        pass

    try:
        MCd["image_standardize"] = MC["data"]["image"]["standardize"]
    except KeyError:
        pass

    return MCd
