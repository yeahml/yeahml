from yeahml.config.helper import parse_yaml_from_path


def extract_hidden_dict_and_set_defaults(m_config: dict) -> dict:
    # TODO: check that the config has only approved values
    # create architecture config
    if m_config["hidden"]["yaml"]:
        h_config = parse_yaml_from_path(m_config["hidden"]["yaml"])
    else:
        # hidden is defined in the current yaml
        # TODO: this needs error checking/handling, empty case
        h_config = m_config["hidden"]
    return h_config
