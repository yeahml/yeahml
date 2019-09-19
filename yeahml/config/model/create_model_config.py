import os

import numpy as np

from yeahml.config.model.components.data import parse_data
from yeahml.config.model.components.hyper_parameters import parse_hyper_parameters

# components
from yeahml.config.model.components.overall import parse_overall

# from yeahml.config.components.hidden import parse_hidden


def extract_model_dict_and_set_defaults(MC: dict) -> dict:
    # this is necessary since the YAML structure is likely to change
    # eventually, this may be deleted

    # TODO: do type checking here... and convert all strings to lower

    main_cdict = {}

    # https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression

    # overall
    overal_dict = parse_overall(MC)
    main_cdict = {**main_cdict, **overal_dict}

    # data
    data_dict = parse_data(MC)
    main_cdict = {**main_cdict, **data_dict}

    # hyperparameters
    hyper_param_dict = parse_hyper_parameters(MC)
    main_cdict = {**main_cdict, **hyper_param_dict}

    return main_cdict


# TODO: will need to implement preprocessing logic
