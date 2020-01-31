import math


class default_config:
    def __init__(self, required=None, description=None, fn=None):
        if required is None:
            self.required = True
        else:
            self.required = False

        if description is None:
            self.description = "The description for this entry hasn't been written yet"
        else:
            self.description = description

        if fn is None:
            self.fn = None
        else:
            self.fn = fn


class numeric(default_config):
    def __init__(
        self,
        required=None,
        description=None,
        value=None,
        bounds=None,
        is_int=None,
        fn=None,
    ):
        super().__init__(required, description, fn)

        if value is None:
            self.value = 42
        else:
            self.value = value

        if bounds is None:
            self.bounds = []
        else:
            self.bounds = (-math.inf, math.inf)

        if is_int is None:
            self.is_int = False
        else:
            self.is_int = is_int

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class categorical:
    def __init__(self, is_subset=None):
        if is_subset is None:
            self.is_subset = []
        else:
            self.wordList = wordList
