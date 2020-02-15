import math

from yeahml.build.components.dtype import return_available_dtypes
from yeahml.build.layers.config import return_layer_defaults, return_available_layers

# TODO: this should be moved
from yeahml.build.layers.config import NOTPRESENT

# TODO: I think the key that is called by this type should be included such that
# we can more easily keep track of where the error is coming from


class default_config:
    def __init__(
        self,
        default_value,
        is_type=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
    ):
        if default_value is None:
            self.default_value = None
        else:
            self.default_value = default_value

        if is_type is None:
            self.is_type = None
        else:
            self.is_type = is_type

        # default to True
        if required is None:
            self.required = True
        else:
            self.required = required

        if description is None:
            self.description = "The description for this entry hasn't been written yet"
        else:
            self.description = description

        if fn is None:
            self.fn = None
        else:
            self.fn = fn

        if fn_args is None:
            self.fn_args = None
        else:
            self.fn_args = fn_args

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    # I don't think this is possible, like I originally thought
    # def __call__(self, cur_value):
    #     print(cur_value)
    #     if self.is_type:
    #         if isinstance(type(cur_value), self.is_type):
    #             raise TypeError(
    #                 f"{cur_value} is (type: {type(cur_value)}) not type {self.is_type}"
    #             )


class numeric(default_config):
    def __init__(
        self,
        default_value=None,
        is_type=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        # specific
        bounds=None,
    ):
        super().__init__(
            default_value=default_value,
            is_type=is_type,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
        )

        if bounds is None:
            self.bounds = (-math.inf, math.inf)
        else:
            # check that bounds are numbers and tuple
            self.bounds = bounds

    def __call__(self, cur_value=None):
        if cur_value and self.is_type:
            if not isinstance(cur_value, self.is_type):
                raise TypeError(
                    f"{cur_value} is (type: {type(cur_value)}) not type {self.is_type}"
                )

        if self.required and not cur_value:
            raise ValueError(f"value was not specified, but is required")

        val = self.default_value
        if cur_value:
            val = cur_value

        if val:
            # ensure w/in bounds
            if val > self.bounds[1]:
                raise ValueError(f"value {val} is greater than {self.bounds[1]}")

            if val < self.bounds[0]:
                raise ValueError(f"value {val} is less than {self.bounds[0]}")
            # TODO: call function with args

        return val


class categorical(default_config):
    def __init__(
        self,
        default_value=None,
        is_type=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        # specific
        is_in_list=None,
        to_lower=None,
    ):
        super().__init__(
            default_value=default_value,
            is_type=is_type,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
        )

        if is_in_list is None:
            self.is_in_list = None
        else:
            self.is_in_list = is_in_list

        # confirm is bool?
        if to_lower is None:
            self.to_lower = True
        else:
            self.to_lower = to_lower

    # call fn with fn_args
    def __call__(self, cur_value=None):
        if cur_value and self.is_type:
            if not isinstance(cur_value, self.is_type):
                raise TypeError(
                    f"{cur_value} is (type: {type(cur_value)}) not type {self.is_type}"
                )

        if self.required and not cur_value:
            raise ValueError(f"value was not specified, but is required")

        val = self.default_value
        if cur_value:
            # NOTE: convert all to lowercase
            val = cur_value

        if self.to_lower and val:
            val = val.lower()

        if val:
            # TODO: call function with args
            if self.is_in_list:
                if val not in self.is_in_list:
                    raise ValueError(f"value {val} is not in {self.is_in_list}")

        return val


class list_of_categorical(categorical):
    def __init__(
        self,
        default_value=None,
        is_type=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        is_in_list=None,
        to_lower=None,
        # specific
        list_must_include=None,
    ):
        super().__init__(
            default_value=default_value,
            is_type=is_type,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
            is_in_list=is_in_list,
            to_lower=to_lower,
        )
        if list_must_include is None:
            self.list_must_include = None
        else:
            self.list_must_include = list_must_include

    def __call__(self, cur_values_list=None):
        if isinstance(cur_values_list, list):
            out_list = []
            # duplicate logic adopted from
            # https://stackoverflow.com/questions/9835762/how-do-i-find-the-duplicates-in-a-list-and-create-another-list-with-them
            duplicates = [
                v
                for i, v in enumerate(cur_values_list)
                if v in cur_values_list[:i] and v is not None
            ]
            if duplicates:
                raise ValueError(f"{duplicates} are duplicated in f{cur_values_list}")

            if self.list_must_include:
                req_not_included = [
                    x for x in self.list_must_include if x not in cur_values_list
                ]
                if req_not_included:
                    raise ValueError(
                        f"{req_not_included} are required but not included in {cur_values_list}"
                    )

            for val in cur_values_list:
                o = categorical(
                    default_value=self.default_value,
                    is_type=self.is_type,
                    required=self.required,
                    description=self.description,
                    fn=self.fn,
                    fn_args=self.fn_args,
                    is_in_list=self.is_in_list,
                )(val)
                out_list.append(o)
        else:
            out_list = [
                categorical(
                    default_value=self.default_value,
                    is_type=self.is_type,
                    required=self.required,
                    description=self.description,
                    fn=self.fn,
                    fn_args=self.fn_args,
                    is_in_list=self.is_in_list,
                )(cur_values_list)
            ]

        return out_list


class optional_config:
    def __init__(self, conf_dict=None):
        if conf_dict is None:
            self.conf_dict = None
        else:
            self.conf_dict = conf_dict

    # def __str__(self):
    #     return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.conf_dict


class parameter_config:
    def __init__(self, known_dict=None, unknown_dict=None):
        if known_dict is None:
            self.known_dict = None
        else:
            self.known_dict = known_dict

        # if unknown_dict is None:
        #     self.unknown_dict = None
        # else:
        #     self.unknown_dict = unknown_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    # def __call__(self):
    #     # out_dict = {**known_dict, **unknown_dict}
    #     out_dict = self.known_dict
    #     return out_dict


class data_in_spec:
    def __init__(self, shape=None, dtype=None):

        self.shape = list_of_numeric(default_value=None, is_type=int, required=True)(
            shape
        )

        self.dtype = categorical(
            default_value=None,
            required=True,
            is_type=str,
            is_in_list=return_available_dtypes(),
        )(dtype)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return {"shape": self.shape, "dtype": self.dtype}


class dict_of_data_in_spec(data_in_spec):
    def __init__(self, required=None):

        if required is None:
            self.required = None
        else:
            self.required = required

    def __call__(self, data_spec_dict):
        if self.required and not data_spec_dict:
            raise ValueError("data_spec_dict is required but not specified")

        out_dict = {}
        if isinstance(data_spec_dict, dict):
            for k, d in data_spec_dict.items():
                out_dict[k] = data_in_spec(shape=d["shape"], dtype=d["dtype"])()
        else:
            raise ValueError(
                f"{data_in_spec} is type {type(data_in_spec)} not type {type({})}"
            )
        return out_dict


class list_of_numeric(numeric):
    def __init__(
        self,
        default_value=None,
        is_type=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        bounds=None,
    ):
        super().__init__(
            default_value=None,
            is_type=None,
            required=None,
            description=None,
            fn=None,
            fn_args=None,
            bounds=None,
        )

    def __call__(self, cur_values_list=None):
        if isinstance(cur_values_list, list):
            out_list = []
            for val in cur_values_list:
                o = numeric(
                    default_value=self.default_value,
                    is_type=self.is_type,
                    required=self.required,
                    description=self.description,
                    fn=self.fn,
                    fn_args=self.fn_args,
                    bounds=self.bounds,
                )(val)
                out_list.append(o)
        else:
            out_list = [
                numeric(
                    default_value=self.default_value,
                    is_type=self.is_type,
                    required=self.required,
                    description=self.description,
                    fn=self.fn,
                    fn_args=self.fn_args,
                    bounds=self.bounds,
                )(cur_values_list)
            ]

        return out_list


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


# class NOTPRESENT:
#     def __init__(self):
#         pass
from yeahml.build.components.activation import _configure_activation
from yeahml.build.components.constraint import _configure_constraint
from yeahml.build.components.initializer import _configure_initializer
from yeahml.build.components.regularizer import _configure_regularizer


SPECIAL_OPTIONS = [
    ("kernel_regularizer", _configure_regularizer),
    ("bias_regularizer", _configure_regularizer),
    ("activity_regularizer", _configure_regularizer),
    ("activation", _configure_activation),
    ("kernel_initializer", _configure_initializer),
    ("bias_initializer", _configure_initializer),
    ("kernel_constraint", _configure_constraint),
    ("bias_constraint", _configure_constraint),
]


class layer_options_config:
    def __init__(self, func_args=None, func_defaults=None, user_args=None):
        # ensure user args is a subset of func_args
        self.user_vals = []
        if user_args:
            special_options_names = [v[0] for v in SPECIAL_OPTIONS]
            for i, arg_name in enumerate(func_args):
                # there are 'special' options... these are parameters that
                # accept classes/functions
                if arg_name in special_options_names:
                    ind = special_options_names.index(arg_name)
                    special_func = SPECIAL_OPTIONS[ind][1]
                    if arg_name in user_args:
                        arg_v = user_args[arg_name]
                        if isinstance(arg_v, str):
                            # make sure string is allowed type
                            pass
                        elif isinstance(arg_v, dict):
                            # do the special function
                            pass
                        else:
                            raise TypeError(
                                f"unable to create {arg_name} with options {arg_v}"
                            )

                if arg_name in user_args:
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
    def __init__(self, layer_type=None, layer_options=None, layer_in_config=None):

        self.layer_base = layer_base_config(layer_type)()
        self.layer_options = layer_options_config(
            func_args=self.layer_base["func_args"],
            func_defaults=self.layer_base["func_defaults"],
            user_args=layer_options,
        )()
        # self.layer_input = layer_input_config(layer_in_config)

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
        if isinstance(model_spec_dict, dict):
            for k, d in model_spec_dict.items():
                out_dict[k] = layer_config(
                    layer_type=d["layer_type"],
                    layer_options=d["layer_options"],
                    layer_in_config=d["in_name"],
                )()
        else:
            raise ValueError(
                f"{data_in_spec} is type {type(data_in_spec)} not type {type({})}"
            )
        return out_dict
