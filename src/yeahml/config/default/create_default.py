from collections import namedtuple
from yeahml.config.default.config_types import numeric, categorical


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


# Data
data = {"data": {}}

dataset = {"dataset": {}}
hyper_parameters = {"hyper_parameters": {}}


logging = {"logging": {}}


model = {"model": {}}

performance = {"performance": {}}

dataset = {"dataset": {}}

DEFAULT_CONFIG = {}
DEFAULT_CONFIG["meta"] = meta

