class Optimizer:
    def __init__(self, optimizer_name, optimizer, obj_dict):
        # this is very confusing and needs to be cleaned up such that it matches
        # the OptimizerWrapper/Optimizers class in train/setup/optimizers.py

        self.name = None if not optimizer_name else optimizer_name
        self.object = optimizer.object

        opt_obj_dict = {}
        for obj_name in optimizer.objectives:
            objective_object = obj_dict[obj_name]
            objective_object.set_optimizer(self)
            opt_obj_dict[obj_name] = objective_object
        self.objectives = opt_obj_dict  # # {obj_a: Objective(), obj_b: Objective()}

        # presently, this is by optimizer, but it might need to be optimizer by
        # ds/in_config to support different shapes/parameters
        # TODO: support more than supervised
        self.get_grads_fn = optimizer.calc_gradient_fn
        self.apply_grads_fn = optimizer.apply_gradient_fn
        self.validation_step_fn = optimizer.inference_fn
        self.losses = optimizer.losses
        self.metrics = optimizer.metrics
        self.num_train_steps = optimizer.num_train_steps

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
