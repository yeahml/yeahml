# TODO
from yeahml.build.components.optimizer import (
    return_available_optimizers,
    return_optimizer,
)


def parse_hyper_parameters(MC: dict) -> dict:
    main_cdict = {}
    main_cdict["epochs"] = MC["hyper_parameters"]["epochs"]
    main_cdict["batch_size"] = MC["hyper_parameters"]["batch_size"]

    # optimizers
    # TODO: this logic below could also be abstracted out
    avail_optimizers = return_available_optimizers()
    try:
        main_cdict["optimizer_dict"] = MC["hyper_parameters"]["optimizer"]
    except KeyError:
        raise KeyError(
            "No optimizer provided: please provide an optimizer to '['hyper_parameters']['optimizer']' in the model config"
        )
    try:
        opts = main_cdict["optimizer_dict"]
    except KeyError:
        pass
        # TODO: no options set -- defulating to X
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

    main_cdict["def_act"] = MC["hyper_parameters"]["default_activation"]
    main_cdict["shuffle_buffer"] = MC["hyper_parameters"]["shuffle_buffer"]

    try:
        main_cdict["early_stopping_e"] = MC["hyper_parameters"]["early_stopping"][
            "epochs"
        ]
    except KeyError:
        # default behavior is to not have early stopping
        # TODO: Log information - default early_stopping_e set to 0
        main_cdict["early_stopping_e"] = 0

    # NOTE: warm_up_epochs is only useful when early_stopping_e > 0
    try:
        main_cdict["warm_up_epochs"] = MC["hyper_parameters"]["early_stopping"][
            "warm_up_epochs"
        ]
    except KeyError:
        # default behavior is to have a warm up period of 5 epochs
        # TODO: Log information - default warm_up_epochs set to 5
        main_cdict["warm_up_epochs"] = 5
    return main_cdict
