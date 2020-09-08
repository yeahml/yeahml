# TODO: implement optimizer
from yeahml.build.components.callback import return_available_callbacks
from yeahml.config.default.types.base_types import categorical, list_of_dict


class callback_config:
    def __init__(self, cb_type=None, cb_options=None):

        # TODO: there are consistency issues here with the names of classes
        # and where the types are being created/checked

        self.type = categorical(
            default_value=None,
            required=True,
            is_in_list=return_available_callbacks(),
            to_lower=True,
            description=(
                "The type of callback being used\n"
                " > e.g. callbacks:objects:'name':type: 'terminateonnan'"
            ),
        )(cb_type)

        self.options = list_of_dict(default_value=None, is_type=list, required=False)(
            cb_options
        )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class callbacks_parser:
    def __init__(self):
        pass

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, callbacks_spec_dict):
        # TODO: this should be moved to the __init__
        if isinstance(callbacks_spec_dict, dict):
            temp_dict = {}
            for k, d in callbacks_spec_dict.items():

                callback_name = categorical(
                    default_value=None,
                    required=True,
                    is_type=str,
                    to_lower=False,
                    description=(
                        "The name of the callback \n"
                        " > e.g. callbacks:objects: 'terminate_on_nan'"
                    ),
                )(k)

                try:
                    callback_type = d["type"]
                except:
                    callback_type = None

                try:
                    callback_options = d["options"]
                except:
                    callback_options = None

                conf = callback_config(
                    cb_type=callback_type, cb_options=callback_options
                )()
                temp_dict[k] = conf

        else:
            raise ValueError(
                f"callbacks_spec_dict ({callbacks_spec_dict}) is type {type(callbacks_spec_dict)} not type {type({})}"
            )
        return temp_dict
