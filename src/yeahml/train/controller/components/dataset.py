class Dataset:
    def __init__(self, name, obj_dict):
        # self.objectives = None  # {obj_a: Objective(), obj_b: Objective()}
        self.name = name
        self.obj_dict = obj_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    # def convert_to_iter?

    # TODO: these could be implemented here -- as well as basic stats kept?
    # def reinit():
    #     pass

    # def take():
    #     pass
