from yeahml.config.default.util import parse_default


def format_performance_config(raw_config, DEFAULT):

    # print(raw_config)

    formatted_dict = {}
    formatted_dict = parse_default(raw_config, DEFAULT)
    # if (
    #     raw_config["type"] == "classification"
    #     or raw_config["type"] == "segmentation"
    # ):
    #     try:
    #         formatted_dict["class_weights"] = np.asarray(raw_config["class_weights"])
    #     except KeyError:
    #         formatted_dict["class_weights"] = np.asarray([1.0] * formatted_dict["num_classes"])

    # convert to lowercase for consistency
    # formatted_dict["loss_fn"] = raw_config["loss_fn"]
    # # TODO: make sure loss_fn is allowed

    # try:
    #     met_list = raw_config["type"]
    #     met_list = [m.lower() for m in met_list]

    #     # TODO: check that metrics are ok/allowed
    #     # convert to set to remove duplication
    #     # met_set = set(met_list)
    # except KeyError:
    #     met_list = None
    #     # raise ValueError("No metrics specified")
    # try:
    #     met_opts_list = raw_config["options"]
    # except KeyError:
    #     met_opts_list = None
    # formatted_dict["met_list"] = met_list
    # formatted_dict["met_opts_list"] = met_opts_list

    return formatted_dict
