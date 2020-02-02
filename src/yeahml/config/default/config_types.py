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

        if val:
            # ensure w/in bounds
            if val > self.bounds[1]:
                raise ValueError(f"value {val} is greater than {self.bounds[1]}")

            if val < self.bounds[0]:
                raise ValueError(f"value {val} is less than {self.bounds[0]}")

            # ensure value is an int if required
            if self.is_int:
                if not isinstance(val, int):
                    raise ValueError(f"value {val} is not a type int type:{type(val)}")

            # TODO: call function with args

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
        is_in_list=None,
    ):
        super().__init__(
            default_value=default_value,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
        )

        if is_in_list is None:
            self.is_in_list = None
        else:
            self.is_in_list = is_in_list

    # call fn with fn_args
    def __call__(self, cur_value=None):
        if self.required and not cur_value:
            raise ValueError(f"value was not specified, but is required")

        val = self.default_value
        if cur_value:
            # NOTE: convert all to lowercase
            val = cur_value.lower()

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
        required=None,
        description=None,
        fn=None,
        fn_args=None,
        is_in_list=None,
        # specific
        list_must_include=None,
    ):
        super().__init__(
            default_value=default_value,
            required=required,
            description=description,
            fn=fn,
            fn_args=fn_args,
            is_in_list=is_in_list,
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

    def __call__(self):
        # out_dict = {**known_dict, **unknown_dict}
        out_dict = known_dict
        return out_dict
