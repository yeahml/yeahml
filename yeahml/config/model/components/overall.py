import numpy as np
import os
import sys
from yeahml.config.helper import create_standard_dirs


def parse_overall(MC: dict) -> dict:
    MCd = {}
    MCd["name"] = MC["overall"]["name"]
    MCd["experiment_dir"] = MC["overall"]["experiment_dir"]
    try:
        MCd["seed"] = MC["overall"]["rand_seed"]
    except KeyError:
        pass

    try:
        MCd["trace_level"] = MC["overall"]["trace"].lower()
    except KeyError:
        pass

    # TODO: this is a temp+new object in the dict
    try:
        MCd["num_classes"] = MC["overall"]["num_classes"]
    except KeyError:
        # no params will be loaded from previously trained params
        # TODO: I don't feel great about this.. this is a temp fix
        if MC["overall"]["metrics"]["type"] == "regression":
            MCd["num_classes"] = 1
        pass

    # if (
    #     MC["overall"]["metrics"]["type"] == "classification"
    #     or MC["overall"]["metrics"]["type"] == "segmentation"
    # ):
    #     try:
    #         MCd["class_weights"] = np.asarray(MC["overall"]["class_weights"])
    #     except KeyError:
    #         MCd["class_weights"] = np.asarray([1.0] * MCd["num_classes"])

    ### architecture
    # TODO: implement after graph can be created...
    MCd["save_params"] = MC["overall"]["saver"]["save_params_name"]

    try:
        MCd["load_params_path"] = MC["overall"]["saver"]["load_params_path"]
    except KeyError:
        # no params will be loaded from previously trained params
        pass

    # convert to lowercase for consistency
    MCd["loss_fn"] = MC["overall"]["loss_fn"].lower()
    # TODO: make sure loss_fn is allowed

    try:
        met_list = MC["overall"]["metrics"]

        # TODO: check that metrics are ok/allowed
        # convert to set to remove duplication
        met_set = set(met_list)
    except KeyError:
        met_set = None
        # raise ValueError("No metrics specified")
    MCd["met_set"] = met_set

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
    #         MCd["metrics_type"] = temp_met_type
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
    # if MCd["metrics_type"] == "classification":
    #     met_set = set(["auc", "accuracy", "precision", "recall"])
    # elif MCd["metrics_type"] == "regression":
    #     met_set = set(["rmse", "mae"])
    # elif MCd["metrics_type"] == "segmentation":
    #     met_set = set(["iou"])
    # else:
    #     # although the error should be caught in the config. the exit error
    #     # is kept until the supported types are pulled from in a config file
    #     # rather than being hardcoded as a list in config.py
    #     sys.exit("metrics type {} is unsupported".format(MCd["metrics_type"]))
    # MCd["met_set"] = met_set

    ####### Logging
    ERR_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    ## console
    try:
        temp_c_lvl = MC["overall"]["logging"]["console"]["level"]
        if not temp_c_lvl:
            # handle null case
            MCd["log_c_lvl"] = "CRITICAL"
        else:
            temp_c_lvl = temp_c_lvl.upper()
            if temp_c_lvl not in ERR_LEVELS:
                sys.exit(
                    "console level {} not allowed. please select one of {}".format(
                        temp_c_lvl, ERR_LEVELS
                    )
                )
            else:
                MCd["log_c_lvl"] = temp_c_lvl
    except KeyError:
        MCd["log_c_lvl"] = "CRITICAL"
        pass

    try:
        MCd["log_c_str"] = MC["overall"]["logging"]["console"]["format_str"]
        if not MCd["log_c_str"]:
            # handle null case
            MCd["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        MCd["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## file
    try:
        temp_f_lvl = MC["overall"]["logging"]["file"]["level"]
        if not temp_f_lvl:
            # handle null case
            MCd["log_f_lvl"] = "CRITICAL"
        else:
            temp_f_lvl = temp_f_lvl.upper()
            if temp_f_lvl not in ERR_LEVELS:
                sys.exit(
                    "console level {} not allowed. please select one of {}".format(
                        temp_f_lvl, ERR_LEVELS
                    )
                )
            else:
                MCd["log_f_lvl"] = temp_f_lvl.upper()
    except KeyError:
        MCd["log_f_lvl"] = "CRITICAL"

    try:
        MCd["log_f_str"] = MC["overall"]["logging"]["file"]["format_str"]
        if not MCd["log_f_str"]:
            # handle null case
            MCd[
                "log_f_str"
            ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        MCd[
            "log_f_str"
        ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"

    ## graph level
    try:
        temp_g_lvl = MC["overall"]["logging"]["graph_spec"]
        if temp_g_lvl == True:
            # support for a simple bool config
            MCd["log_g_lvl"] = "DEBUG"
        elif not temp_g_lvl:
            # handle null case
            MCd["log_g_lvl"] = "DEBUG"
        else:
            temp_g_lvl = temp_g_lvl.upper()
            if temp_g_lvl not in ERR_LEVELS:
                sys.exit(
                    "log level {} not allowed. please select one of {}".format(
                        temp_g_lvl, ERR_LEVELS
                    )
                )
            else:
                MCd["log_g_lvl"] = temp_g_lvl
    except KeyError:
        MCd["log_g_lvl"] = "DEBUG"
        pass
    # hard set the graph info
    MCd["log_g_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## preds level
    try:
        temp_p_lvl = MC["overall"]["logging"]["graph_spec"]
        if temp_p_lvl == True:
            # support for a simple bool config
            MCd["log_p_lvl"] = "DEBUG"
        elif not temp_p_lvl:
            # handle null case
            MCd["log_p_lvl"] = "DEBUG"
        else:
            temp_p_lvl = temp_p_lvl.upper()
            if temp_p_lvl not in ERR_LEVELS:
                sys.exit(
                    "log level {} not allowed. please select one of {}".format(
                        temp_p_lvl, ERR_LEVELS
                    )
                )
            else:
                MCd["log_p_lvl"] = temp_p_lvl
    except KeyError:
        MCd["log_p_lvl"] = "DEBUG"
        pass
    # hard set the graph info
    MCd["log_p_str"] = "[%(levelname)-8s] %(message)s"

    # DEV_DIR is a hardcoded value for the directory in which the examples
    # are located. for packing+, this will need to be removed.
    DEV_DIR = "examples"
    # BEST_PARAMS_DIR is a hardcoded value that must match the created dir in
    # create_standard_dirs
    BEST_PARAMS_DIR = "best_params"
    MCd["log_dir"] = os.path.join(
        ".", DEV_DIR, MC["overall"]["name"], MCd["experiment_dir"]
    )
    MCd["save_weights_path"] = os.path.join(
        ".",
        DEV_DIR,
        MC["overall"]["name"],
        MCd["experiment_dir"],
        BEST_PARAMS_DIR,
        MCd["save_params"] + ".h5",  # TODO: modify
    )
    # wipe is set to true for now
    create_standard_dirs(MCd["log_dir"], True)

    return MCd
