from yeahml.build.components.activation import configure_activation
from yeahml.build.components.constraint import configure_constraint
from yeahml.build.components.initializer import configure_initializer
from yeahml.build.components.regularizer import configure_regularizer

# TODO: this should be moved
from yeahml.build.layers.config import (
    NOTPRESENT,
    return_available_layers,
    return_layer_defaults,
)
from yeahml.config.default.types.base_types import (
    categorical,
    default_config,
    list_of_categorical,
    list_of_numeric,
    numeric,
)
from yeahml.config.default.types.compound.data import data_in_spec

SPECIAL_OPTIONS = [
    ("kernel_regularizer", configure_regularizer),
    ("bias_regularizer", configure_regularizer),
    ("activity_regularizer", configure_regularizer),
    ("activation", configure_activation),
    ("kernel_initializer", configure_initializer),
    ("bias_initializer", configure_initializer),
    ("kernel_constraint", configure_constraint),
    ("bias_constraint", configure_constraint),
]


class layer_base_config:
    def __init__(self, layer_type=None):
        if layer_type is None:
            raise ValueError("layer_type is not defined")
        else:
            self.str = categorical(
                required=True, is_type=str, is_in_list=return_available_layers().keys()
            )(layer_type)
            fn_dict = return_layer_defaults(self.str)
            # {"func": func, "func_args": func_args, "func_defaults": func_defaults}
            self.func = fn_dict["func"]
            self.func_args = fn_dict["func_args"]
            self.func_defaults = fn_dict["func_defaults"]

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class layer_options_config:
    def __init__(self, func_args=None, func_defaults=None, user_args=None):
        # ensure user args is a subset of func_args
        self.user_vals = []
        if user_args:
            special_options_names = [v[0] for v in SPECIAL_OPTIONS]
            for i, arg_name in enumerate(func_args):
                # there are 'special' options... these are parameters that
                # accept classes/functions
                if arg_name in user_args:
                    if arg_name in special_options_names:
                        # arg_v = None
                        ind = special_options_names.index(arg_name)
                        special_func = SPECIAL_OPTIONS[ind][1]
                        if arg_name in user_args:
                            arg_v = user_args[arg_name]
                            if isinstance(arg_v, dict):
                                # extract useful components
                                try:
                                    func_type = arg_v["type"]
                                except KeyError:
                                    raise ValueError(
                                        "Function for {arg_name} is not specified as a type:[<insert_type_here>]"
                                    )
                                try:
                                    func_opts = arg_v["options"]
                                except KeyError:
                                    func_opts = None

                                # use special functions
                                # TODO: make sure the special_func accepts this signature
                                arg_v = special_func(func_type, func_opts)
                            else:
                                raise TypeError(
                                    f"unable to create {arg_name} with options {arg_v} - a dictionary (with :type) is required"
                                )
                    else:
                        arg_v = user_args[arg_name]
                else:
                    arg_v = func_defaults[i]
                    if type(arg_v) == type(NOTPRESENT):
                        raise ValueError(
                            f"arg value for {arg_name} is not specified, but is required to be specified"
                        )
                self.user_vals.append(arg_v)
            # sanity check
            assert len(self.user_vals) == len(
                func_defaults
            ), f"user vals not set correctly"
        else:
            # no options are specified, but some arguments require it
            for i, arg_default in enumerate(func_defaults):
                if type(arg_default) == type(NOTPRESENT):
                    raise ValueError(
                        f"arg value for {func_args[i]} is not specified, but is required to be specified"
                    )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class layer_config:
    def __init__(self, layer_type=None, layer_options=None, layer_in_name=None):

        self.layer_base = layer_base_config(layer_type)()
        self.layer_options = layer_options_config(
            func_args=self.layer_base["func_args"],
            func_defaults=self.layer_base["func_defaults"],
            user_args=layer_options,
        )()
        self.layer_in_name = categorical(
            default_value=None, required=True, is_type=str
        )(layer_in_name)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class layers_config:
    def __init__(self, conf_dict=None):
        if conf_dict is None:
            self.conf_dict = None
        else:
            self.conf_dict = conf_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, model_spec_dict):
        out_dict = {}
        prev_layer_name = None
        if isinstance(model_spec_dict, dict):
            for k, d in model_spec_dict.items():
                try:
                    layer_in_name = d["in_name"]
                except KeyError:
                    layer_in_name = prev_layer_name
                out_dict[k] = layer_config(
                    layer_type=d["layer_type"],
                    layer_options=d["layer_options"],
                    layer_in_name=layer_in_name,
                )()
                # increment layer
                prev_layer_name = k
        else:
            raise ValueError(
                f"{data_in_spec} is type {type(data_in_spec)} not type {type({})}"
            )

        return out_dict
