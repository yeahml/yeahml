# directory strcuture: ./yeahml_dir/name/experiment_name

# default
# required: True
# -- checking
# > numeric
# bounds {upper: , lower: }
# is_int: True
# > categorical
# issubset ["", ""]
# description -- string

META = {
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


data = {"data": {}}

hyper_parameters = {"hyper_parameters": {}}

logging = {"logging": {}}


model = {"model": {}}

performance = {"performance": {}}

dataset = {"dataset": {}}
