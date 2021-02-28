from yeahml.build.components.optimizer import configure_optimizer
from yeahml.train.gradients.gradients import (
    get_apply_grad_fn,
    get_get_supervised_grads_fn,
    get_validation_step_fn,
)


class OptimizerWrapper:
    def __init__(
        self,
        name,
        tf_object,
        objectives,
        losses,
        metrics,
        calc_gradient_fn,
        apply_gradient_fn,
        inference_fn,
        num_train_steps=0,
    ):
        self.name = name
        self.object = tf_object
        self.objectives = objectives
        # used to determine which objectives to loop to calculate losses
        self.losses = losses
        # used to determine which objectives to obtain to calculate metrics
        self.metrics = metrics
        # create a tf.function for applying gradients for each optimizer
        # TODO: I am not 100% about this logic for maping the optimizer to the
        #   apply_gradient fn... this needs to be confirmed to work as expected
        self.calc_gradient_fn = calc_gradient_fn
        self.apply_gradient_fn = apply_gradient_fn
        self.inference_fn = inference_fn
        self.num_train_steps = num_train_steps


class Optimizers:
    """
    give each optimizer an:
        - gradient calc function
        - apply gradient functions
        - inference functions
        - book keeping (number of optimization steps)

    maybe needs:
        - mapping of optimizer to loss objectives and metric objectives
    
    """

    def __init__(self, optim_cdict, objectives_dict):
        optimizers_dict = {}
        for opt_name, opt_dict in optim_cdict["optimizers"].items():
            configured_optimizer = configure_optimizer(opt_dict)

            loss_objective_names = []
            metrics_objective_names = []
            for cur_objective in opt_dict["objectives"]:
                cur_objective_dict = objectives_dict[cur_objective]
                if "loss" in cur_objective_dict.keys():
                    if cur_objective_dict["loss"]:
                        loss_objective_names.append(cur_objective)
                if "metrics" in cur_objective_dict.keys():
                    if cur_objective_dict["metrics"]:
                        metrics_objective_names.append(cur_objective)

            optimizers_dict[opt_name] = OptimizerWrapper(
                name=opt_name,
                tf_object=configured_optimizer,
                objectives=opt_dict["objectives"],
                losses=loss_objective_names,
                metrics=metrics_objective_names,
                calc_gradient_fn=get_get_supervised_grads_fn(),
                apply_gradient_fn=get_apply_grad_fn(),
                inference_fn=get_validation_step_fn(),
            )
        self.optimizers = optimizers_dict
