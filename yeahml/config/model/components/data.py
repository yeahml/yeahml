# TODO


def parse_data(MC: dict) -> dict:
    main_cdict = {}
    ## inputs
    # TODO: this is sloppy...
    # copy is used to prevent overwriting underlying data
    main_cdict["input_layer_dim"] = None
    main_cdict["in_dim"] = MC["data"]["in"]["dim"].copy()
    if main_cdict["in_dim"][0]:  # as oppposed to [None, x, y, z]
        main_cdict["in_dim"].insert(0, None)  # add batching
    main_cdict["in_dtype"] = MC["data"]["in"]["dtype"]
    try:
        main_cdict["reshape_in_to"] = MC["data"]["in"]["reshape_to"]
        if main_cdict["reshape_in_to"][0] != -1:  # as oppposed to [None, x, y, z]
            main_cdict["reshape_in_to"].insert(0, -1)  # -1
    except KeyError:
        # None in this case is representative of not reshaping
        main_cdict["reshape_in_to"] = None
    if main_cdict["reshape_in_to"]:
        main_cdict["input_layer_dim"] = MC["data"]["in"]["reshape_to"]
    else:
        main_cdict["input_layer_dim"] = MC["data"]["in"]["dim"].copy()

    # copy is used to prevent overwriting underlying data
    main_cdict["output_dim"] = MC["data"]["label"]["dim"].copy()
    if main_cdict["output_dim"][0]:  # as oppposed to [None, x, y, z]
        main_cdict["output_dim"].insert(0, None)  # add batching
    main_cdict["label_dtype"] = MC["data"]["label"]["dtype"]

    try:
        temp_one_hot = MC["data"]["label"]["one_hot"]
        if temp_one_hot != False and temp_one_hot != True:
            raise ValueError(
                f"Error > Exiting: data:label:one_hot {temp_one_hot} unsupported. Please use True or False"
            )
        main_cdict["label_one_hot"] = temp_one_hot
    except KeyError:
        # None in this case is representative of not using one hot encoding
        main_cdict["label_one_hot"] = False

    # currently required
    main_cdict["TFR_dir"] = MC["data"]["TFR"]["dir"]
    main_cdict["TFR_train"] = MC["data"]["TFR"]["train"]
    main_cdict["TFR_test"] = MC["data"]["TFR"]["test"]
    main_cdict["TFR_val"] = MC["data"]["TFR"]["validation"]

    # TODO: this is a first draft for this type of organization and will
    # will likely be changed
    main_cdict["data_in_dict"] = MC["data"]["in"]
    main_cdict["data_out_dict"] = MC["data"]["label"]
    main_cdict["TFR_parse"] = MC["data"]["TFR_parse"]

    try:
        main_cdict["augmentation"] = MC["data"]["image"]["augmentation"]
    except KeyError:
        pass

    try:
        main_cdict["image_standardize"] = MC["data"]["image"]["standardize"]
    except KeyError:
        pass

    return main_cdict
