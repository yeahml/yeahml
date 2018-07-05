#!/usr/bin/env python
import yaml
import shutil
import os
import sys
import logging
from yf_logging import config_logger


def parse_yaml_from_path(path: str) -> dict:
    # return python dict from yaml path
    try:
        with open(path, "r") as stream:
            try:
                y = yaml.load(stream)
                return y
            except yaml.YAMLError as exc:
                print(exc)
                return dict()
    except FileNotFoundError:
        sys.exit(
            "Error > Exiting: the model configuration file {} was not found".format(
                path
            )
        )


def create_model_and_hidden_config(path: str) -> tuple:
    # return the model and archtitecture configuration dicts
    m_config = parse_yaml_from_path(path)
    if not m_config:
        sys.exit(
            "Error > Exiting: the model config file was found {}, but appears to be empty".format(
                path
            )
        )
    # create architecture config
    if m_config["hidden"]["yaml"]:
        h_config = parse_yaml_from_path(m_config["hidden"]["yaml"])
    else:
        # hidden is defined in the current yaml
        # TODO: this needs error checking/handling, empty case
        h_config = m_config["hidden"]

    m_config, h_config = extract_dict_and_set_defaults(m_config, h_config)

    return (m_config, h_config)


# helper to create dirs if they don't already exist
def maybe_create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("{} created".format(dir_path))
    else:
        print("{} already exists".format(dir_path))


def create_standard_dirs(root_dir: str, wipe_dirs: bool):
    # this logic is messy
    if wipe_dirs:
        if os.path.exists(root_dir):
            shutil.rmtree(root_dir)
        maybe_create_dir(root_dir)
    else:
        maybe_create_dir(root_dir)

    # maybe_create_dir(root_dir + "/saver")
    # `best_params/` will hold a serialized version of the best params
    # I like to keep this as a backup in case I run into issues with
    # the saver files
    maybe_create_dir(root_dir + "/best_params")
    # `tf_logs/` will hold the logs that will be visible in tensorboard
    maybe_create_dir(root_dir + "/tf_logs")

    # `yf_logs/` will hold the custom logs
    maybe_create_dir(root_dir + "/yf_logs")


def extract_dict_and_set_defaults(MC: dict, HC: dict) -> tuple:
    # this is necessary since the YAML structure is likely to change
    # eventually, this may be deleted

    # TODO: do type checking here... and convert all strings to lower

    MCd = {}

    MCd["experiment_dir"] = MC["overall"]["experiment_dir"]

    ## inputs
    MCd["in_dim"] = MC["data"]["in"]["dim"]
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

    MCd["output_dim"] = MC["data"]["label"]["dim"]
    if MCd["output_dim"][0]:  # as oppposed to [None, x, y, z]
        MCd["output_dim"].insert(0, None)  # add batching
    MCd["label_dtype"] = MC["data"]["label"]["dtype"]

    # currently required
    MCd["TFR_dir"] = MC["data"]["TFR"]["dir"]
    MCd["TFR_train"] = MC["data"]["TFR"]["train"]
    MCd["TFR_test"] = MC["data"]["TFR"]["test"]
    MCd["TFR_val"] = MC["data"]["TFR"]["validation"]

    ########### hyperparameters
    MCd["lr"] = MC["hyper_parameters"]["lr"]
    MCd["epochs"] = MC["hyper_parameters"]["epochs"]
    MCd["batch_size"] = MC["hyper_parameters"]["batch_size"]
    MCd["optimizer"] = MC["hyper_parameters"]["optimizer"]
    MCd["def_act"] = MC["hyper_parameters"]["default_activation"]
    MCd["shuffle_buffer"] = MC["hyper_parameters"]["shuffle_buffer"]

    ## validation
    try:
        MCd["early_stopping_e"] = MC["hyper_parameters"]["early_stopping"]["epochs"]
    except KeyError:
        # default behavior is to not have early stopping
        # TODO: Log information - default early_stopping_e set to 0
        MCd["early_stopping_e"] = 0

    # NOTE: warm_up_epochs is only useful when early_stopping_e > 0
    try:
        MCd["warm_up_epochs"] = MC["hyper_parameters"]["early_stopping"][
            "warm_up_epochs"
        ]
    except KeyError:
        # default behavior is to have a warm up period of 5 epochs
        # TODO: Log information - default warm_up_epochs set to 5
        MCd["warm_up_epochs"] = 5

    ### architecture
    # TODO: implement after graph can be created...
    MCd["save_params"] = MC["overall"]["saver"]["save_params_name"]
    MCd["load_params_path"] = MC["overall"]["saver"]["load_params_path"]

    MCd["final_type"] = MC["overall"]["type"]
    try:
        MCd["seed"] = MC["overall"]["rand_seed"]
    except KeyError:
        pass

    try:
        MCd["trace_level"] = MC["overall"]["trace"]
    except KeyError:
        pass

    MCd["print_g_spec"] = MC["overall"]["print_graph_spec"]
    MCd["name"] = MC["overall"]["name"]

    try:
        MCd["augmentation"] = MC["train"]["augmentation"]
    except KeyError:
        pass

    try:
        MCd["image_standardize"] = MC["train"]["image_standardize"]
    except KeyError:
        pass

    # TODO: this may instead be handled by the log level, e.g. if <= DEBUG
    try:
        MCd["full_error"] = MC["overall"]["full_error_message"]
    except KeyError:
        MCd["full_error"] = False

    # DEV_DIR is a hardcoded value for the directory in which the examples
    # are located. for packing+, this will need to be removed.
    DEV_DIR = "example"
    # BEST_PARAMS_DIR is a hardcoded value that must match the created dir in
    # create_standard_dirs
    BEST_PARAMS_DIR = "best_params"
    MCd["log_dir"] = os.path.join(
        ".", DEV_DIR, MC["overall"]["name"], MCd["experiment_dir"]
    )
    MCd["saver_save"] = os.path.join(
        ".",
        DEV_DIR,
        MC["overall"]["name"],
        MCd["experiment_dir"],
        BEST_PARAMS_DIR,
        MCd["save_params"] + ".ckpt",
    )
    # wipe is set to true for now
    create_standard_dirs(MCd["log_dir"], True)

    ####### Logging
    # console
    ERR_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
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

    # file
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
            MCd["log_f_str"] = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        MCd["log_f_str"] = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

    # set up logger
    # config_logger(MCd)

    return (MCd, HC)


# TODO: will need to implement preprocessing logic
