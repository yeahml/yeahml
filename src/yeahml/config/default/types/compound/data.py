from yeahml.build.components.dtype import return_available_dtypes
from yeahml.config.default.types.base_types import categorical, list_of_numeric

# TODO: I think the key that is called by this type should be included such that
# we can more easily keep track of where the error is coming from


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
