from yeahml.dataset.dataset import Dataset


class DatasetWrapper:
    def __init__(self, name, obj_dict):
        # self.objectives = None  # {obj_a: Objective(), obj_b: Objective()}
        self.name = name
        self.dataset = Dataset(name, obj_dict)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
