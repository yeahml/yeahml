from yeahml.train.gradients.gradients import (
    get_apply_grad_fn,
    get_get_supervised_grads_fn,
    get_validation_step_fn,
)

"""

Attempting to codify the relationships that could exist

left to right: datasets -->  objectives -->  optimizers

that is, objectives use multiple datasets, but are optimized by a single
optimizer. Optimizers can optimize multiple objectives. objectives can learn
from multiple datasets. Each ojective has a performance object with at least 1
loss and N metrics

"""


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


class Performance:
    # NOTE: what if we want to keep track of metrics that don't rely on the
    # current head/output graph? (would need an outter metric)
    def __init__(self, loss_config, metric_config):
        self.loss = loss_config["object"]  # single loss
        self.metric_dict = metric_config

        # losses and metrics are different in tf. that is losses don't have
        # internal state, whereas metrics do

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Objective:
    def __init__(self, name, config, ds_dict):

        # NOTE: there may not be an optimizer here if this is only a metric?
        self.optimizer = None  # Optimizer()

        self.type = config["in_config"]["type"]
        self.prediction = config["in_config"]["options"]["prediction"]
        self.target = (
            None
            if not config["in_config"]["options"]["target"]
            else config["in_config"]["options"]["target"]
        )
        # self.datasets = None  # {ds_a: Dataset(), ds_b: Dataset()}
        self.dataset = ds_dict[config["in_config"]["dataset"]]

        if "loss" in config.keys():
            l_conf = config["loss"]
        else:
            l_conf = None

        if "metrics" in config.keys():
            m_conf = config["metrics"]
        else:
            m_conf = None

        self.performance = Performance(l_conf, m_conf)

        # right now, we're allowing only one dataset.. but, I think in the
        # future, we need to think through how this should define our
        # nomenclature a bit better. that is, is an objective a task, does a
        # task have multiple objectives?

    def set_optimizer(self, opt_object):
        if not self.optimizer:
            self.optimizer = opt_object
        else:
            raise ValueError(
                f"optimizer previously set as {self.optimizer}, cannot overwrite to {opt_name}"
            )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


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


class Controller:
    def __init__(
        self,
        optimizers_dict,
        dataset_dict,
        objectives_dict,
        obj_policy=None,
        training=False,
    ):
        """[summary]

        Parameters
        ----------
        optimizers_dict : [type]
            [description]
        dataset_dict : [type]
            [description]
        objectives_dict : [type]
            [description]
        obj_policy : [type], optional
            a policy that decides which objective to select, by default None
        training : bool, optional
            [description], by default False
        """
        self.training = training

        self.objectives = None  # {obj_a: Objective(), obj_b: Objective()}
        self.cur_obj = None
        self.obj_policy = obj_policy

        self.datasets = None  # {ds_a: Dataset(), ds_b: Dataset()}
        self.cur_ds = None

        self.optimizers = None  # {opt_a: Optimizer(), opt_b: Optimizer()}
        self.cur_optimizer = None

        ds_dict = {}
        for cur_ds_name, cur_ds_conf in dataset_dict.items():
            ds_dict[cur_ds_name] = Dataset(cur_ds_name, cur_ds_conf)
        self.datasets = ds_dict

        obj_dict = {}
        for cur_obj_name, cur_obj_conf in objectives_dict.items():
            obj_object = Objective(cur_obj_name, cur_obj_conf, ds_dict)
            obj_dict[cur_obj_name] = obj_object
        self.objectives = obj_dict

        opt_dict = {}
        for cur_opt_name, cur_opt_conf in optimizers_dict.items():
            opt_object = Optimizer(cur_opt_name, cur_opt_conf, obj_dict)
            opt_dict[cur_opt_name] = opt_object
        self.optimizers = opt_dict

        # TODO: validate all connections ds --> obj --> optimizer ?

        self._initialize(self.obj_policy)

    def _initialize(self, obj_policy):
        # TODO: implement `initialize` to allow setting the initial information

        if not obj_policy:
            obj_policy = self.obj_policy

        first_obj = self.select_objective(obj_policy=obj_policy)
        self.set_by_objective(first_obj)

    def select_objective(self, obj_policy=None):
        if obj_policy:
            raise NotImplementedError(
                "selecting the objective based on a policy is not implemented yet"
            )
        else:
            # select first
            obj = list(self.objectives.keys())[0]

        return obj

    def set_by_objective(self, obj_name):
        self.cur_objective = self.objectives[obj_name]
        self.cur_dataset = self.objectives[obj_name].dataset
        self.cur_optimizer = self.objectives[obj_name].optimizer

    # TODO:
    # def select_by_______a________
    # def select_by_______b________
    # I'd like to see more methods here for selecting which objective/optimizer
    # to select

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# TODO: validate connections?
