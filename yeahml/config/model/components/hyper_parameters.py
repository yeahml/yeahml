# TODO


def parse_hyper_parameters(MC: dict) -> dict:
    MCd = {}
    MCd["lr"] = MC["hyper_parameters"]["lr"]
    MCd["epochs"] = MC["hyper_parameters"]["epochs"]
    MCd["batch_size"] = MC["hyper_parameters"]["batch_size"]
    MCd["optimizer"] = MC["hyper_parameters"]["optimizer"]
    MCd["def_act"] = MC["hyper_parameters"]["default_activation"]
    MCd["shuffle_buffer"] = MC["hyper_parameters"]["shuffle_buffer"]

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
    return MCd
