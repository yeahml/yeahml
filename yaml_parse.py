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


# TODO: will need to implement preprocessing logic
