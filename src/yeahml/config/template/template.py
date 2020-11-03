from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi
from yeahml.build.components.optimizer import return_available_optimizers
import crummycm as ccm

"""
TODO: 
- rethink naming, headings, etc....
- descriptions
- not sure what to do about `epochs`

"""


meta = {
    "meta": {
        KPH("yeahml_dir", exact=True, required=False): Text(
            default_value="yeahml",
            description=(
                "Root directory to store information\n"
                " > e.g. meta:yeahml_dir: 'yeahml'"
            ),
        ),
        "data_name": Text(
            description=(
                "Description of the data used \n"
                " > e.g. meta:data_name: 'mnist', or  meta:data_name: 'V00'\n"
                "this logic will likely change in the future"
            )
        ),
        "experiment_name": Text(
            description=(
                "Name for the experiment being performed\n"
                " > e.g. meta:experiment_name: 'trial_00'"
            )
        ),
        "start_fresh": Bool(
            default_value=False,
            description=(
                "Used to determine whether previous experiments should be deleted\n"
                " > e.g. meta:start_fresh: True"
            ),
        ),
        KPH("random_seed", exact=True, required=False): Numeric(
            is_type=int,
            description=(
                "Used to set the random seed for tensorflow\n"
                " > e.g. meta:rand_seed: 42"
            ),
        ),
        KPH("default_load_params_path", exact=True, required=False): Text(),
        # TODO: tracing
        # "trace_level": Text(description="meta:trace_level: <str>"),
    }
}


hyper_parameters = {
    "hyper_parameters": {
        "dataset": {
            "batch": Numeric(
                is_type=int, description="hyper_parameters:dataset:batch: <int>"
            ),
            KPH("shuffle_buffer", exact=True, required=False): Numeric(
                is_type=int,
                description="hyper_parameters:dataset:shuffle_buffer: <int>",
            ),
        },
        "epochs": Numeric(is_type=int, description="hyper_parameters:epochs: <int>"),
        KPH("early_stopping", exact=True, required=False): {
            "epochs": Numeric(is_type=int, description="patience"),
            "warm_up": Numeric(
                is_type=int,
                description="allow training to 'warm up' before keeping track",
            ),
        },
    }
}

optimize = {
    "optimize": {
        "optimizers": {
            KPH("optimizer_name", multi=True): {
                "type": Text(
                    is_in_list=return_available_optimizers(),
                    to_lower=True,
                    description=(
                        "The type of optimizer being used\n"
                        " > e.g. optimize:optimizers:'name':type: 'adam'"
                    ),
                ),
                "options": {
                    "learning_rate": Numeric(is_type=float),
                    KPH("other_options"): VPH("optimizer_options"),
                },
                "objectives": VPH("optimizer_objectives"),
            }
        }
        # "directive": {"instructions": "SS"},
    }
}

ERR_LEVELS = [x.lower() for x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]

logging = {
    "logging": {
        "console": {
            "level": Text(
                default_value="critical",
                is_in_list=ERR_LEVELS,
                to_lower=True,
                description="level to log information to the console",
            ),
            "format_str": Text(
                default_value="%(name)-12s: %(levelname)-8s %(message)s"
            ),
        },
        "file": {
            "level": Text(
                default_value="critical",
                is_in_list=ERR_LEVELS,
                to_lower=True,
                description="level to log information to a file",
            ),
            "format_str": Text(
                default_value="%(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"
            ),
        },
        "track": {
            "tracker_steps": Numeric(
                default_value=0,
                is_type=int,
                description="the frequency (as a number of training steps) at which to log tracker information",
            ),
            "tensorboard": {"param_steps": Numeric(is_type=int)},
        },
    }
}

data = {
    "data": {
        "datasets": {
            KPH("dataset_name"): {
                "in": {
                    KPH("feat_name"): {
                        "shape": Multi(),
                        "dtype": Text(),
                        "startpoint": Bool(),
                        "endpoint": Bool(),
                        "label": Bool(),
                    }
                },
                "split": {"names": Multi(element_types=Text())},
            }
        }
    }
}

performance = {
    "performance": {
        "objectives": {
            KPH("objective_name"): {
                "loss": {
                    "type": Text(),
                    "options": VPH("loss_options"),
                    "track": VPH("loss_track"),
                },
                "metric": {
                    "type": Text(),
                    "options": VPH("loss_options"),
                    "track": VPH("loss_track"),
                },
                "in_config": {
                    "type": Text(),
                    "options": {"prediction": Text(), "target": Text()},
                    "dataset": Text(),
                },
            }
        }
    }
}

model = {
    "model": {
        "name": Text(),
        "start_fresh": Bool(),
        "layers": {
            KPH("layer_name"): {
                "type": Text(),
                "source": Text(),
                "options": {
                    KPH("layer_option_key", multi=True): VPH("layer_option_value"),
                    KPH("activation", exact=True, required=False): {
                        "type": Text(),
                        KPH("options_key", required=False, multi=True): VPH(
                            "options_value"
                        ),
                    },
                },
            }
        },
    }
}

callbacks = {
    "callbacks": {
        "objects": {
            KPH("callback_name", multi=True): {
                "type": Text(to_lower=True),
                "options": {
                    KPH("options_key", multi=True, required=False): VPH("options_value")
                },
            }
        }
    }
}

TEMPLATE = {}
TEMPLATE = {**TEMPLATE, **meta}
TEMPLATE = {**TEMPLATE, **performance}
TEMPLATE = {**TEMPLATE, **hyper_parameters}
TEMPLATE = {**TEMPLATE, **logging}
TEMPLATE = {**TEMPLATE, **data}
TEMPLATE = {**TEMPLATE, **model}
TEMPLATE = {**TEMPLATE, **optimize}
TEMPLATE = {**TEMPLATE, **callbacks}



