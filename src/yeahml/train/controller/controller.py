from yeahml.train.controller.components.dataset import DatasetWrapper
from yeahml.train.controller.components.objective import Objective
from yeahml.train.controller.components.optimizer import Optimizer


"""

Attempting to codify the relationships that could exist

left to right: datasets -->  objectives -->  optimizers

that is, objectives use multiple datasets, but are optimized by a single
optimizer. Optimizers can optimize multiple objectives. objectives can learn
from multiple datasets. Each ojective has a performance object with at least 1
loss and N metrics

"""


class DSNode:
    def __init__(self, name):
        self.name = name
        self.objectives = None

    def add_objective(self, objective):
        if not self.objectives:
            self.objectives = [objective]  # presently only 1
        else:
            self.objectives.append(objective)


class ObjectiveNode:
    def __init__(self, name):
        self.name = name
        self.datasets = None
        self.optimizer = None  # presently only be 1

    def add_dataset(self, dataset):
        if not self.datasets:
            self.datasets = [dataset]
        else:
            self.datasets.append(dataset)

    def add_optimizer(self, optimizer):
        if not self.optimizer:
            self.optimizer = optimizer
        else:
            raise ValueError(
                f"optimizer already exists {self.optimizer}, can't add {optimizer}"
            )


class OptimizerNode:
    def __init__(self, name):
        self.name = name
        self.objectives = None

    def add_objective(self, objective):
        if not self.objectives:
            self.objectives = [objective]
        else:
            self.objectives.append(objective)


class ControllerRelationships:
    def __init__(self, optimizers_dict, dataset_dict, objectives_dict):
        opt_node_dict = {}
        for opt_name, _ in optimizers_dict:
            opt_node_dict[opt_name] = OptimizerNode(opt_name)

        ds_node_dict = {}
        for ds_name, _ in dataset_dict:
            ds_node_dict[ds_name] = DSNode(ds_name)

        obj_node_dict = {}
        for obj_name, _ in objectives_dict:
            obj_node_dict[obj_name] = DSNode(obj_name)

        self.objectives = objectives_dict
        self.datasets = ds_node_dict
        self.objectives = obj_node_dict

        # TODO: make connections

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


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

        self.datasets = None  # {ds_a: DatasetWrapper(), ds_b: DatasetWrapper()}
        self.cur_ds = None

        self.optimizers = None  # {opt_a: Optimizer(), opt_b: Optimizer()}
        self.cur_optimizer = None

        ds_dict = {}
        for cur_ds_name, cur_ds_conf in dataset_dict.items():
            ds_dict[cur_ds_name] = DatasetWrapper(cur_ds_name, cur_ds_conf)
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
