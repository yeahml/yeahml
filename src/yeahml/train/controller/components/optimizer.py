from yeahml.train.gradients.gradients import (
    get_apply_grad_fn,
    get_get_supervised_grads_fn,
    get_validation_step_fn,
)


class Optimizer:
    def __init__(self, optimizer_name, optimizer_config, obj_dict):
        self.name = None if not optimizer_name else optimizer_name
        self.object = optimizer_config["optimizer"]

        opt_obj_dict = {}
        for obj_name in optimizer_config["objectives"]:
            objective_object = obj_dict[obj_name]
            objective_object.set_optimizer(self)
            opt_obj_dict[obj_name] = objective_object
        self.objectives = opt_obj_dict  # # {obj_a: Objective(), obj_b: Objective()}

        # presently, this is by optimizer, but it might need to be optimizer by
        # ds/in_config to support different shapes/parameters
        # TODO: support more than supervised
        self.get_grads_fn = get_get_supervised_grads_fn()
        self.apply_grads_fn = get_apply_grad_fn()
        self.validation_step_fn = get_validation_step_fn()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
