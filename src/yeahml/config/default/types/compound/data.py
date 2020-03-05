from yeahml.build.components.dtype import return_available_dtypes
from yeahml.config.default.types.base_types import categorical, list_of_numeric

# TODO: I think the key that is called by this type should be included such that
# we can more easily keep track of where the error is coming from

# TODO: the data spec should mirror the layer spec for input layers..


class data_in_spec:
    def __init__(self, shape=None, dtype=None, startpoint=True, endpoint=False):

        self.shape = list_of_numeric(default_value=None, is_type=int, required=True)(
            shape
        )

        self.dtype = categorical(
            default_value=None,
            required=True,
            is_type=str,
            is_in_list=return_available_dtypes(),
        )(dtype)

        self.startpoint = startpoint
        self.endpoint = endpoint

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


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

                try:
                    layer_is_startpoint = d["startpoint"]
                except KeyError:
                    layer_is_startpoint = True

                try:
                    layer_is_endpoint = d["endpoint"]
                except KeyError:
                    layer_is_endpoint = False

                try:
                    out_dict[k] = data_in_spec(
                        shape=d["shape"],
                        dtype=d["dtype"],
                        startpoint=layer_is_startpoint,
                        endpoint=layer_is_endpoint,
                    )()
                except TypeError:
                    raise TypeError(
                        f"item [key={k}:dict={d}] was not expected. in the key:dict, the dict is expected to have subkeys :shape and :dtype. full spec: {data_spec_dict}"
                    )
        else:
            raise ValueError(
                f"{data_in_spec} is type {type(data_in_spec)} not type {type({})}"
            )
        return out_dict
