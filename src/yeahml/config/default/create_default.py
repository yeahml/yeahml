from collections import namedtuple
from config_types import numeric


def hithere():
    return 42


a = numeric()
print(a)
b = numeric(value=22, fn=hithere)
print(b)

# loop keys
# if type == dict recursive
# else, fill
# return

# meta
meta = {
    "meta": {
        # directory
        "yeahml_dir": {"default": "yeahml", "required": False},
        "data_name": {"default": None, "required": True},
        "experiment_name": {"default": None, "required": False},
        # random seed
        "rand_seed": {"default": None, "required": False},
        # tracing
        "trace_level": {"default": None, "required": False},
        # default path to load param information
        # TODO: this should likely move to the model config
        "default_load_params_path": {"default": None, "required": False},
    }
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

