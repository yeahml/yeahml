def format_performance_config(raw_config):
    formatted_dict = {}

    # TODO: this is a temp+new object in the dict
    # try:
    #     formatted_dict["num_classes"] = raw_config["num_classes"]
    # except KeyError:
    #     # no params will be loaded from previously trained params
    #     # TODO: I don't feel great about this.. this is a temp fix
    #     if raw_config["type"] == "regression":
    #         formatted_dict["num_classes"] = 1
    #     pass

    # if (
    #     raw_config["type"] == "classification"
    #     or raw_config["type"] == "segmentation"
    # ):
    #     try:
    #         formatted_dict["class_weights"] = np.asarray(raw_config["class_weights"])
    #     except KeyError:
    #         formatted_dict["class_weights"] = np.asarray([1.0] * formatted_dict["num_classes"])

    # convert to lowercase for consistency
    formatted_dict["loss_fn"] = raw_config["loss_fn"]
    # TODO: make sure loss_fn is allowed

    try:
        met_list = raw_config["type"]
        met_list = [m.lower() for m in met_list]

        # TODO: check that metrics are ok/allowed
        # convert to set to remove duplication
        # met_set = set(met_list)
    except KeyError:
        met_list = None
        # raise ValueError("No metrics specified")
    try:
        met_opts_list = raw_config["options"]
    except KeyError:
        met_opts_list = None
    formatted_dict["met_list"] = met_list
    formatted_dict["met_opts_list"] = met_opts_list

    # ## "type" of problem (will set the default performance metrics)
    # try:
    #     # TODO: these types+options should come from a config
    #     METRIC_TYPES = ["classification", "regression", "segmentation"]
    #     temp_met_type = raw_config["type"]
    #     temp_met_type = temp_met_type.lower()
    #     if temp_met_type not in METRIC_TYPES:
    #         sys.exit(
    #             "metric type {} not allowed. please select one of {}".format(
    #                 temp_met_type, METRIC_TYPES
    #             )
    #         )
    #     else:
    #         formatted_dict["metrics_type"] = temp_met_type
    # except:
    #     sys.exit(
    #         "overall:metrics:type: was not specified. please select one of {}".format(
    #             METRIC_TYPES
    #         )
    #     )
    # # set default metrics for the specified type
    # # - available metrics
    # # > "tn", "tp", "fn", "fp"
    # # > "accuracy", "precision", "recall", "auc"
    # # > "rmse", "mae"
    # # > "iou"
    # if formatted_dict["metrics_type"] == "classification":
    #     met_set = set(["auc", "accuracy", "precision", "recall"])
    # elif formatted_dict["metrics_type"] == "regression":
    #     met_set = set(["rmse", "mae"])
    # elif formatted_dict["metrics_type"] == "segmentation":
    #     met_set = set(["iou"])
    # else:
    #     # although the error should be caught in the config. the exit error
    #     # is kept until the supported types are pulled from in a config file
    #     # rather than being hardcoded as a list in config.py
    #     sys.exit("metrics type {} is unsupported".format(formatted_dict["metrics_type"]))
    # formatted_dict["met_set"] = met_set

    return formatted_dict
