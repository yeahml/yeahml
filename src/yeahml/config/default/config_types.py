import math


class default_config:
    def __init__(
        self, default_value, required=None, description=None, fn=None, fn_args=None
    ):
        if default_value is None:
            self.default_value = None
        else:
            self.default_value = default_value

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


class numeric(default_config):
    def __init__(
        self,
        default_value=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        # specific
        bounds=None,
        is_int=None,
    ):
        super().__init__(
            default_value=default_value,
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

        if is_int is None:
            self.is_int = False
        else:
            self.is_int = is_int

    def __call__(self, cur_value=None):
        if self.required and not cur_value:
            raise ValueError(f"value was not specified, but is required")

        val = self.default_value
        if cur_value:
            val = cur_value

        # ensure w/in bounds
        # if init_value > self.bounds[1]:
        #     raise ValueError(f"value {init_value} is greater than ")
        return val


class categorical(default_config):
    def __init__(
        self,
        default_value=None,
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        # specific
        is_subset=None,
    ):
        super().__init__(
            default_value=default_value,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
        )

        if is_subset is None:
            self.is_subset = []
        else:
            self.is_subset = is_subset

    # call fn with fn_args
    def __call__(self, cur_value=None):
        if self.required and not cur_value:
            raise ValueError(f"value was not specified, but is required")

        val = self.default_value
        if cur_value:
            val = cur_value

        return val
