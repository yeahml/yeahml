from yeahml.config.default.types.base_types import categorical, numeric
from yeahml.config.default.types.compound.callbacks import callbacks_parser
from yeahml.config.default.types.compound.data import data_set_name_dict
from yeahml.config.default.types.compound.directive import instruct_parser
from yeahml.config.default.types.compound.layer import layers_parser
from yeahml.config.default.types.compound.optimizer import optimizers_parser
from yeahml.config.default.types.compound.performance import performances_parser
from yeahml.config.default.types.param_types import optional_config


# TODO: check for extra keys in the configs that are not specified here
# meta
# TODO: set accepted options for `trace_level`
# TODO: ensure `default_load_params_path` is a path.. also, does this belong in
# meta?
# TODO: numbers could probably be converted to string (for experiment_name?)


# TODO: some of these values are positive only .. may consider additional check
hyper_parameters = {
    "hyper_parameters": {
        "dataset": {
            "batch": numeric(
                default_value=None,
                required=True,
                is_type=int,
                description="hyper_parameters:dataset:batch: <int>",
            ),
            "shuffle_buffer": numeric(
                default_value=None,
                required=False,
                is_type=int,
                description="hyper_parameters:dataset:shuffle_buffer: <int>",
            ),
        },
        "epochs": numeric(
            default_value=None,
            required=True,
            is_type=int,
            description="hyper_parameters:epochs: <int>",
        ),
        # TODO: need to account for optional outter keys
        "early_stopping": optional_config(
            conf_dict={
                "epochs": numeric(
                    default_value=None,
                    required=False,
                    is_type=int,
                    description="hyper_parameters:early_stopping:epochs: <int>",
                ),
                "warm_up": numeric(
                    default_value=None,
                    required=False,
                    is_type=int,
                    description="hyper_parameters:early_stopping:warm_up: <int>",
                ),
            }
        ),
    }
}

optimize = {
    "optimize": {
        "optimizers": optimizers_parser(),
        "directive": {"instructions": instruct_parser()},
    }
}


# Data
data = {"data": {"datasets": data_set_name_dict(required=True)}}


# TODO: eventually, we need to support custom performance/loss metrics
performance = {"performance": {"objectives": performances_parser()}}
model = {
    "model": {
        # directory
        # TODO: check that no spaces or special chars are included in the model
        # name and other directory names?
        "name": categorical(
            default_value=None,
            required=True,
            is_type=str,
            description=("name of the model\n" " > e.g. model:name: 'jacks_model"),
        ),
        "start_fresh": categorical(
            default_value=False,
            required=False,
            is_type=bool,
            description=(
                "model start_fresh `start_fresh: <bool>` is used to determine "
                "whether to start the directory 'fresh'/delete current contents \n"
                " > e.g. model:start_fresh: True"
            ),
        ),
        "layers": layers_parser(),  # could add a check that the location exists
    }
}

callbacks = {"callbacks": {"objects": callbacks_parser()}}
