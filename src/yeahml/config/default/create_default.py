from yeahml.build.components.loss import return_available_losses
from yeahml.build.components.metric import return_available_metrics
from yeahml.build.components.optimizer import return_available_optimizers
from yeahml.config.default.config_types import (
    categorical,
    list_of_categorical,
    numeric,
    optional_config,
    parameter_config,
)


def hithere():
    return 42


# numeric(
#     required=None,
#     description=None,
#     fn=None,
#     fn_args=None,
#     # specifc
#     value=None,
#     bounds=None,
#     is_int=None,
# )
# categorical(
#     required=None,
#     description=None,
#     fn=None,
#     fn_args=None,
#     # specific
#     value=None,
#     is_subset=None,
# )
# loop keys
# if type == dict recursive
# else, fill
# return

# meta
meta = {
    "meta": {
        # directory
        "yeahml_dir": categorical(
            default_value="yeahml", required=False
        ),  # could add a check that the location exists
        "data_name": categorical(default_value=None, required=True),
        "experiment_name": categorical(default_value=None, required=True),
        # random seed
        "rand_seed": numeric(default_value=None, required=False),
        # tracing
        "trace_level": categorical(default_value=None, required=False),
        # default path to load param information
        # TODO: this should likely move to the model config
        "default_load_params_path": categorical(
            default_value=None, required=False
        ),  # TODO: confirm path exists
    }
}

# TODO: eventually, we need to support custom performance/loss metrics
performance = {
    "performance": {
        "metric": {
            "type": list_of_categorical(
                default_value=None, required=True, is_in_list=return_available_metrics()
            ),
            "options": list_of_categorical(default_value=None, required=False),
        },
        # TODO: support multiple losses -- currently only one loss is supported
        "loss": {
            "type": categorical(
                default_value=None, required=True, is_in_list=return_available_losses()
            ),
            # TODO: error check that options are valid
            "options": categorical(default_value=None, required=False),
        },
    }
}

hyper_parameters = {
    "hyper_parameters": {
        "dataset": {
            "batch": numeric(default_value=None, required=True, is_int=True),
            "shuffle_buffer": numeric(default_value=None, required=False, is_int=True),
        },
        "epochs": numeric(default_value=None, required=True, is_int=True),
        # TODO: need to account for optional outter keys
        "early_stopping": optional_config(
            conf_dict={
                "epochs": numeric(default_value=None, required=False, is_int=True),
                "warm_up": numeric(default_value=None, required=False, is_int=True),
            }
        ),
        # TODO: Right now I'm assuming 1 loss and one optimizer.. this isn't
        # and needs to be reconsidered
        "optimizer": {
            "type": categorical(
                default_value=None,
                required=True,
                is_in_list=return_available_optimizers(),
            ),
            # TODO: this isn't really a list of categorical.... most are numeric
            "options": parameter_config(
                known_dict={
                    "learning_rate": numeric(
                        default_value=None, required=True, is_int=False
                    )
                }
            ),
        },
    }
}


ERR_LEVELS = [x.lower() for x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]

logging = {
    "logging": {
        "console": {
            "level": categorical(
                default_value="CRITICAL", required=False, is_in_list=ERR_LEVELS
            ),
            "format_str": categorical(
                default_value="%(name)-12s: %(levelname)-8s %(message)s", required=False
            ),
        },
        "file": {
            "level": categorical(
                default_value="CRITICAL", required=False, is_in_list=ERR_LEVELS
            ),
            "format_str": categorical(
                default_value="%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s",
                required=False,
            ),
        },
    }
}


# Data
data = {"data": {}}
dataset = {"dataset": {}}
model = {"model": {}}
dataset = {"dataset": {}}


DEFAULT_CONFIG = {}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **meta}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **performance}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **hyper_parameters}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **logging}
