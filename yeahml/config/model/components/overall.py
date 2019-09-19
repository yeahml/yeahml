import os

import numpy as np

from yeahml.config.helper import create_standard_dirs


def parse_overall(MC: dict) -> dict:
    main_cdict = {}
    main_cdict["name"] = MC["overall"]["name"]
    main_cdict["experiment_dir"] = MC["overall"]["experiment_dir"]
    try:
        main_cdict["seed"] = MC["overall"]["rand_seed"]
    except KeyError:
        pass

    try:
        main_cdict["trace_level"] = MC["overall"]["trace"].lower()
    except KeyError:
        pass

    # TODO: this is a temp+new object in the dict
    # try:
    #     main_cdict["num_classes"] = MC["overall"]["num_classes"]
    # except KeyError:
    #     # no params will be loaded from previously trained params
    #     # TODO: I don't feel great about this.. this is a temp fix
    #     if MC["overall"]["metrics"]["type"] == "regression":
    #         main_cdict["num_classes"] = 1
    #     pass

    # if (
    #     MC["overall"]["metrics"]["type"] == "classification"
    #     or MC["overall"]["metrics"]["type"] == "segmentation"
    # ):
    #     try:
    #         main_cdict["class_weights"] = np.asarray(MC["overall"]["class_weights"])
    #     except KeyError:
    #         main_cdict["class_weights"] = np.asarray([1.0] * main_cdict["num_classes"])

    ### architecture
    # TODO: implement after graph can be created...
    main_cdict["save_params"] = MC["overall"]["saver"]["save_params_name"]

    try:
        main_cdict["load_params_path"] = MC["overall"]["saver"]["load_params_path"]
    except KeyError:
        # no params will be loaded from previously trained params
        pass

    # convert to lowercase for consistency
    main_cdict["loss_fn"] = MC["overall"]["loss_fn"]
    # TODO: make sure loss_fn is allowed

    try:
        met_list = MC["overall"]["metrics"]["type"]
        met_list = [m.lower() for m in met_list]

        # TODO: check that metrics are ok/allowed
        # convert to set to remove duplication
        # met_set = set(met_list)
    except KeyError:
        met_list = None
        # raise ValueError("No metrics specified")
    try:
        met_opts_list = MC["overall"]["metrics"]["options"]
    except KeyError:
        met_opts_list = None
    main_cdict["met_list"] = met_list
    main_cdict["met_opts_list"] = met_opts_list

    # ## "type" of problem (will set the default performance metrics)
    # try:
    #     # TODO: these types+options should come from a config
    #     METRIC_TYPES = ["classification", "regression", "segmentation"]
    #     temp_met_type = MC["overall"]["metrics"]["type"]
    #     temp_met_type = temp_met_type.lower()
    #     if temp_met_type not in METRIC_TYPES:
    #         sys.exit(
    #             "metric type {} not allowed. please select one of {}".format(
    #                 temp_met_type, METRIC_TYPES
    #             )
    #         )
    #     else:
    #         main_cdict["metrics_type"] = temp_met_type
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
    # if main_cdict["metrics_type"] == "classification":
    #     met_set = set(["auc", "accuracy", "precision", "recall"])
    # elif main_cdict["metrics_type"] == "regression":
    #     met_set = set(["rmse", "mae"])
    # elif main_cdict["metrics_type"] == "segmentation":
    #     met_set = set(["iou"])
    # else:
    #     # although the error should be caught in the config. the exit error
    #     # is kept until the supported types are pulled from in a config file
    #     # rather than being hardcoded as a list in config.py
    #     sys.exit("metrics type {} is unsupported".format(main_cdict["metrics_type"]))
    # main_cdict["met_set"] = met_set

    ####### Logging
    ERR_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    ## console
    try:
        temp_c_lvl = MC["overall"]["logging"]["console"]["level"]
        if not temp_c_lvl:
            # handle null case
            main_cdict["log_c_lvl"] = "CRITICAL"
        else:
            temp_c_lvl = temp_c_lvl.upper()
            if temp_c_lvl not in ERR_LEVELS:
                raise ValueError(
                    f"console level {temp_c_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                main_cdict["log_c_lvl"] = temp_c_lvl
    except KeyError:
        main_cdict["log_c_lvl"] = "CRITICAL"
        pass

    try:
        main_cdict["log_c_str"] = MC["overall"]["logging"]["console"]["format_str"]
        if not main_cdict["log_c_str"]:
            # handle null case
            main_cdict["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        main_cdict["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## file
    try:
        temp_f_lvl = MC["overall"]["logging"]["file"]["level"]
        if not temp_f_lvl:
            # handle null case
            main_cdict["log_f_lvl"] = "CRITICAL"
        else:
            temp_f_lvl = temp_f_lvl.upper()
            if temp_f_lvl not in ERR_LEVELS:
                raise ValueError(
                    "console level {temp_f_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                main_cdict["log_f_lvl"] = temp_f_lvl.upper()
    except KeyError:
        main_cdict["log_f_lvl"] = "CRITICAL"

    try:
        main_cdict["log_f_str"] = MC["overall"]["logging"]["file"]["format_str"]
        if not main_cdict["log_f_str"]:
            # handle null case
            main_cdict[
                "log_f_str"
            ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        main_cdict[
            "log_f_str"
        ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"

    ## graph level
    try:
        temp_g_lvl = MC["overall"]["logging"]["graph_spec"]
        if temp_g_lvl == True:
            # support for a simple bool config
            main_cdict["log_g_lvl"] = "DEBUG"
        elif not temp_g_lvl:
            # handle null case
            main_cdict["log_g_lvl"] = "DEBUG"
        else:
            temp_g_lvl = temp_g_lvl.upper()
            if temp_g_lvl not in ERR_LEVELS:
                raise ValueError(
                    "log level {temp_g_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                main_cdict["log_g_lvl"] = temp_g_lvl
    except KeyError:
        main_cdict["log_g_lvl"] = "DEBUG"
        pass
    # hard set the graph info
    main_cdict["log_g_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## preds level
    try:
        temp_p_lvl = MC["overall"]["logging"]["graph_spec"]
        if temp_p_lvl == True:
            # support for a simple bool config
            main_cdict["log_p_lvl"] = "DEBUG"
        elif not temp_p_lvl:
            # handle null case
            main_cdict["log_p_lvl"] = "DEBUG"
        else:
            temp_p_lvl = temp_p_lvl.upper()
            if temp_p_lvl not in ERR_LEVELS:
                raise ValueError(
                    "log level {temp_p_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                main_cdict["log_p_lvl"] = temp_p_lvl
    except KeyError:
        main_cdict["log_p_lvl"] = "DEBUG"
        pass
    # hard set the graph info
    main_cdict["log_p_str"] = "[%(levelname)-8s] %(message)s"

    # DEV_DIR is a hardcoded value for the directory in which the examples
    # are located. for packing+, this will need to be removed.
    DEV_DIR = "examples"
    # BEST_PARAMS_DIR is a hardcoded value that must match the created dir in
    # create_standard_dirs
    BEST_PARAMS_DIR = "best_params"
    main_cdict["log_dir"] = os.path.join(
        ".", DEV_DIR, MC["overall"]["name"], main_cdict["experiment_dir"]
    )
    main_cdict["save_weights_path"] = os.path.join(
        ".",
        DEV_DIR,
        MC["overall"]["name"],
        main_cdict["experiment_dir"],
        BEST_PARAMS_DIR,
        main_cdict["save_params"] + ".h5",  # TODO: modify
    )
    # wipe is set to true for now
    create_standard_dirs(main_cdict["log_dir"], True)

    return main_cdict
