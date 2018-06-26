#!/usr/bin/env python
import yaml


def parse_yaml_from_path(path: str) -> dict:
    # return python dict from yaml path
    with open(path, "r") as stream:
        try:
            y = yaml.load(stream)
            return y
        except yaml.YAMLError as exc:
            print(exc)
            return None


def create_model_and_arch_config(path: str) -> (dict, dict):
    # return the model and archtitecture configuration dicts
    m_config = parse_yaml_from_path(path)
    # create architecture config
    if m_config["architecture"]["yaml"]:
        a_config = parse_yaml_from_path(m_config["architecture"]["yaml"])
    else:
        a_config = m_config["architecture"]

    return (m_config, a_config)


def extract_from_dict(MC: dict, AC: dict) -> (dict, dict):
    # this is necessary since the YAML structure is likely to change
    # eventually, this can be deleted
    MCd = {}
    # ACd = {}
    ## inputs
    MCd["in_dim"] = MC["data"]["in_dim"]
    if MCd["in_dim"][0]:
        MCd["in_dim"].insert(0, None)  # add batching

    MCd["output_dim"] = MC["data"]["output_dim"]
    if MCd["output_dim"][0]:
        MCd["output_dim"].insert(0, None)  # add batching

    MCd["TFR_dir"] = MC["data"]["TFR_dir"]

    ## hyperparams
    MCd["lr"] = MC["hyper_parameters"]["lr"]
    MCd["epochs"] = MC["hyper_parameters"]["epochs"]
    MCd["batch_size"] = MC["hyper_parameters"]["batch_size"]
    ## implementation
    MCd["optimizer"] = MC["implementation"]["optimizer"]
    MCd["def_act"] = MC["implementation"]["default_activation"]
    MCd["shuffle_buffer"] = MC["implementation"]["shuffle_buffer"]

    ### architecture
    # TODO: implement after graph can be created...l
    MCd["save_pparams"] = MC["saver"]["save_pparams"]
    MCd["final_type"] = MC["overall"]["options"]
    MCd["seed"] = MC["overall"]["seed"]

    return (MCd, AC)


# TODO: will need to implement preprocessing logic
