from yeahml.build.components.loss import return_available_losses
from yeahml.build.components.metric import return_available_metrics
from yeahml.config.default.config_types import categorical, list_of_categorical, numeric


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


# Data
data = {"data": {}}
dataset = {"dataset": {}}
hyper_parameters = {"hyper_parameters": {}}
logging = {"logging": {}}
model = {"model": {}}
dataset = {"dataset": {}}


DEFAULT_CONFIG = {}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **meta}
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **performance}
