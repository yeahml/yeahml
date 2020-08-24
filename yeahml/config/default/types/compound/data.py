from yeahml.build.components.dtype import return_available_dtypes
from yeahml.config.default.types.base_types import categorical, list_of_numeric

# TODO: I think the key that is called by this type should be included such that
# we can more easily keep track of where the error is coming from

# TODO: the data spec should mirror the layer spec for input layers..


class data_in_spec:
    def __init__(
        self, shape=None, dtype=None, startpoint=True, endpoint=False, label=False
    ):

        # TODO: this is a bandaid fix and will need to be addressed in a future.
        # I am not currently sure how best to handle this case.
        if shape == "unknown":
            self.shape = (None,)
        else:
            self.shape = list_of_numeric(
                default_value=None,
                is_type=int,
                required=True,
                description=(
                    "shape of the data feature\n"
                    " > e.g. data:datasets:'mnist':in:image_in:shape: [28,28,1]"
                ),
            )(shape)
            # TODO: include name?

        self.dtype = categorical(
            default_value=None,
            required=True,
            is_type=str,
            is_in_list=return_available_dtypes(),
            description=(
                "dtype of the feature\n"
                " > e.g. data:datasets:'mnist':in:image_in:dtype: 'float32"
            ),
        )(dtype)

        self.startpoint = startpoint
        self.endpoint = endpoint
        self.label = label

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


def _parse_split_config(raw):
    try:
        split_names = raw["names"]
    except KeyError:
        raise KeyError(f"no names were provided for current dataset")

    return {"names": split_names}


class dict_of_data_in_spec(data_in_spec):
    # This code is nearly embarassing
    def __init__(self, required=None):

        if required is None:
            self.required = None
        else:
            self.required = required

    def __call__(self, data_spec_dict):

        if self.required and not data_spec_dict:
            raise ValueError("data_spec_dict is required but not specified")

        ####################### parse :in
        try:
            temp_in_dict = data_spec_dict["in"]
        except KeyError:
            raise KeyError(
                f"no :in key was specified for current data config: {data_spec_dict}"
            )

        temp_out_dict = {}
        if isinstance(temp_in_dict, dict):
            for k, d in temp_in_dict.items():

                # TODO: should startpoint/endpoint be checked as bools?
                try:
                    layer_is_startpoint = d["startpoint"]
                except KeyError:
                    layer_is_startpoint = True

                try:
                    layer_is_endpoint = d["endpoint"]
                except KeyError:
                    layer_is_endpoint = False

                try:
                    layer_is_label = d["label"]
                except KeyError:
                    layer_is_label = False

                # if the layer is a label it is assumed (perhaps incorrectly,
                # but we can address this as needed), that the layer is also a endpoint
                if layer_is_label:
                    layer_is_endpoint = True

                try:
                    temp_out_dict[k] = data_in_spec(
                        shape=d["shape"],
                        dtype=d["dtype"],
                        startpoint=layer_is_startpoint,
                        endpoint=layer_is_endpoint,
                        label=layer_is_label,
                    )()
                except TypeError:
                    raise TypeError(
                        f"item [key={k}:dict={d}] was not expected. in the key:dict,"
                        f" the dict is expected to have subkeys :shape and :dtype. full spec: {data_spec_dict}"
                    )
        else:
            raise ValueError(
                f"{temp_in_dict} is type {type(temp_in_dict)} not type {type({})}"
            )

        # parse split information
        # TODO: should this be 'splits' plural?
        try:
            split_config_raw = data_spec_dict["split"]
        except KeyError:
            raise KeyError(
                f"no :split key was specified for current data config: {data_spec_dict}"
            )

        # TODO: more advanced parsing here
        out_split_config = _parse_split_config(split_config_raw)

        out_dict = {"in": temp_out_dict, "split": out_split_config}
        return out_dict


class data_set_name_dict:
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
            for ds_name, ds_dict in data_spec_dict.items():
                parsed_dict = dict_of_data_in_spec(required=True)(ds_dict)
                out_dict[ds_name] = parsed_dict
        else:
            raise ValueError(
                f"{data_spec_dict} is type {type(data_spec_dict)} not type {type({})}"
            )
        return out_dict
