class Dataset:
    def __init__(self, name, obj_dict):
        # self.objectives = None  # {obj_a: Objective(), obj_b: Objective()}
        self.name = name

        iter_dict, split_count = {}, {}
        for split_name, tf_ds in obj_dict.items():
            iter_dict[split_name] = tf_ds.repeat(1).__iter__()
            split_count[split_name] = 0

        self.obj_dict = obj_dict
        self.iter_dict = iter_dict
        self.split_count = split_count

    def get_next_batch(self, split_name):
        ds_iter = self.iter_dict[split_name]
        try:
            batch = next(ds_iter)
        except StopIteration:
            # add count and reinitialize iter
            batch = None
            self.split_count[split_name] += 1
            self.iter_dict[split_name] = self.obj_dict[split_name].repeat(1).__iter__()
        return batch

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

