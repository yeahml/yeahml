# TODO
from yeahml.build.components.optimizer import (
    return_available_optimizers,
    return_optimizer,
)


def _return_data_hp(raw_config):

    # TODO: ACCEPTED = {"batch": {default:None}, "shuffle": {default:None}}

    formatted_dict = {}
    try:
        ds_hp_raw = raw_config["dataset"]
        ds_hp_format = formatted_dict["dataset"] = {}
    except KeyError:
        raise KeyError(
            f"No dataset hyperparameters set in '['hyper_parameters']['dataset']' in the model config"
        )

    try:
        ds_hp_format["batch"] = ds_hp_raw["batch"]
    except KeyError:
        raise KeyError(
            f"No batch_size set in '['hyper_parameters']['dataset']['batch']' in the model config"
        )

    return formatted_dict


def format_hyper_parameters_config(raw_config: dict) -> dict:
    formatted_config = {}

    # formatted_config["epochs"] = raw_config["epochs"]
    # formatted_config["batch_size"] = raw_config["batch_size"]
    tmp_dict = _return_data_hp(raw_config)
    formatted_config = {**formatted_config, **tmp_dict}

    try:
        formatted_config["epochs"] = raw_config["epochs"]
    except KeyError:
        raise KeyError(
            f"No epochs set in '['hyper_parameters']['epochs']' in the model config"
        )

    # optimizers
    # TODO: this logic below could also be abstracted out
    avail_optimizers = return_available_optimizers()
    try:
        formatted_config["optimizer_dict"] = raw_config["optimizer"]
    except KeyError:
        raise KeyError(
            "No optimizer provided: please provide an optimizer to '['hyper_parameters']['optimizer']' in the model config"
        )
    try:
        opts = formatted_config["optimizer_dict"]
    except KeyError:
        pass
        # TODO: no options set -- defaulting to X
    if opts["type"] not in list(avail_optimizers.keys()):
        raise ValueError(
            f"optimizer {opts['type']} not available in {avail_optimizers.keys()}"
        )
    temp_opt = return_optimizer(opts["type"])
    for opt in opts:
        if opt != "type":
            if opt not in temp_opt["func_args"]:
                raise ValueError(
                    f"option {opt} not available; please use one of {opt['func_args']}"
                )

    try:
        formatted_config["early_stopping_e"] = raw_config["early_stopping"]["epochs"]
    except KeyError:
        # default behavior is to not have early stopping
        # TODO: Log information - default early_stopping_e set to 0
        formatted_config["early_stopping_e"] = 0

    # NOTE: warm_up_epochs is only useful when early_stopping_e > 0
    try:
        formatted_config["warm_up_epochs"] = raw_config["early_stopping"][
            "warm_up_epochs"
        ]
    except KeyError:
        # default behavior is to have a warm up period of 5 epochs
        # TODO: Log information - default warm_up_epochs set to 5
        formatted_config["warm_up_epochs"] = 5

    # TODO: try + default
    # try:
    #     formatted_config["warm_up_epochs"] = raw_config["early_stopping"][
    #         "warm_up_epochs"
    #     ]
    # except KeyError:
    #     # default behavior is to have a warm up period of 5 epochs
    #     # TODO: Log information - default warm_up_epochs set to 5
    #     formatted_config["warm_up_epochs"] = 5
    # formatted_config["shuffle_buffer"] = raw_config["shuffle_buffer"]

    return formatted_config
