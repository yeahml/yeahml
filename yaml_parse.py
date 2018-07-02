#!/usr/bin/env python
import yaml
import shutil
import os
import sys


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
    # `tf_logs/` will hold the logs that will be visable in tensorboard
    maybe_create_dir(root_dir + "/tf_logs")


def extract_dict_and_set_defaults(MC: dict, HC: dict) -> tuple:
    # this is necessary since the YAML structure is likely to change
    # eventually, this may be deleted
    MCd = {}
    ## inputs
    MCd["in_dim"] = MC["data"]["in"]["dim"]
    if MCd["in_dim"][0]:  # as oppposed to [None, x, y, z]
        MCd["in_dim"].insert(0, None)  # add batching
    MCd["in_dtype"] = MC["data"]["in"]["dtype"]
    try:
        MCd["reshape_in_to"] = MC["data"]["in"]["reshape_to"]
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
    MCd["save_pparams"] = MC["saver"]["save_pparams"]
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

    MCd["log_dir"] = os.path.join(
        ".", "example", "cats_v_dogs_01", MC["tensorboard"]["log_dir"]
    )
    # wipe is set to true for now
    create_standard_dirs(MCd["log_dir"], True)

    return (MCd, HC)


# TODO: will need to implement preprocessing logic
