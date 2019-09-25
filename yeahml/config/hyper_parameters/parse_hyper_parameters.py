# TODO
from yeahml.build.components.optimizer import (
    return_available_optimizers,
    return_optimizer,
)


def format_hyper_parameters_config(raw_config: dict) -> dict:
    formatted_config = {}
    formatted_config["epochs"] = raw_config["epochs"]
    formatted_config["batch_size"] = raw_config["batch_size"]

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

    formatted_config["def_act"] = raw_config["default_activation"]
    formatted_config["shuffle_buffer"] = raw_config["shuffle_buffer"]

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
    return formatted_config
