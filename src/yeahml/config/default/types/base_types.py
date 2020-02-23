import math


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