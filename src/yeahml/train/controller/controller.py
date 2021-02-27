import random

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

TODO (in semi-order):
1. clean file
2. create the appropriate select_by_xxxx methods
3. decide what a ``policy'' should look like
4. tracker/statistics access
5. model access


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

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ObjectiveNode:
    def __init__(self, name):
        self.name = name
        self.datasets = None
        self.optimizer = None  # presently only be 1

        # NOTE: these three could be grouped
        self.type = None
        self.prediction = None
        self.target = None

    def add_type_information(self, in_config):
        if self.type:
            raise ValueError(
                f"type has already been set to {self.type} cannot set: {in_config}"
            )
        else:
            self.type = in_config["type"]
            if self.type == "supervised":
                self.prediction_node = in_config["options"]["prediction"]
                self.target_node = in_config["options"]["target"]
            else:
                self.prediction = in_config["options"]["prediction"]

    def add_datasets(self, dataset):
        if isinstance(dataset, str):
            dataset = [dataset]
        if not isinstance(dataset, list):
            raise ValueError(
                f"dataset {dataset} is of unexpected type. expected str/list, got {type(dataset)}"
            )
        if not self.datasets:
            self.datasets = dataset
        else:
            self.datasets += dataset

    def add_optimizer(self, optimizer):
        if not self.optimizer:
            self.optimizer = optimizer
        else:
            raise ValueError(
                f"optimizer already exists {self.optimizer}, can't add {optimizer}"
            )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class OptimizerNode:
    def __init__(self, name):
        self.name = name
        self.objectives = None

    def add_objective(self, objective):
        if isinstance(objective, str):
            objective = [objective]
        if not isinstance(objective, list):
            raise ValueError(
                f"objective {objective} is of unexpected type. expected str/list, got {type(objective)}"
            )
        if not self.objectives:
            self.objectives = objective
        else:
            self.objectives += objective

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ControllerRelationships:
    def __init__(self, optimizers_dict, dataset_dict, objectives_dict):

        # build node skeleton
        opt_node_dict = {}
        for opt_name, _ in optimizers_dict.items():
            opt_node_dict[opt_name] = OptimizerNode(opt_name)

        ds_node_dict = {}
        for ds_name, _ in dataset_dict.items():
            ds_node_dict[ds_name] = DSNode(ds_name)

        obj_node_dict = {}
        for obj_name, _ in objectives_dict.items():
            obj_node_dict[obj_name] = ObjectiveNode(obj_name)

        # build connections (probably could be clever and merge above and this
        # loop if performance was an issue)
        for obj_name, obj_node in obj_node_dict.items():
            obj_node.add_type_information(objectives_dict[obj_name]["in_config"])
            obj_node.add_datasets(objectives_dict[obj_name]["in_config"]["dataset"])
            tmp_ds = objectives_dict[obj_name]["in_config"]["dataset"]
            if isinstance(tmp_ds, str):
                tmp_ds = [tmp_ds]
            if not isinstance(tmp_ds, list):
                raise ValueError(f"{tmp_ds} is not type list or str")
            for ds in tmp_ds:
                ds_node_dict[ds].add_objective(obj_name)

            # TODO: add information to DS

        # TODO: loop optimizers and add objectives
        for opt_name, opt_node in opt_node_dict.items():
            tmp_objs = optimizers_dict[opt_name]["objectives"]
            opt_node.add_objective(tmp_objs)
            if isinstance(tmp_objs, str):
                tmp_objs = [tmp_objs]
            if not isinstance(tmp_objs, list):
                raise ValueError(f"{tmp_objs} is not type list or str")
            for to in tmp_objs:
                obj_node_dict[to].add_optimizer(opt_name)

        self.optimizers = opt_node_dict
        self.datasets = ds_node_dict
        self.objectives = obj_node_dict

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
        """

        I think the controller needs access to the tracker/statistics about each
        dataset/objective/optimizer such that it can make ``informed'' decisions
        as well as information about the model/params
        """
        self.training = training

        self.objectives = None  # {obj_a: Objective(), obj_b: Objective()}
        self.objectives_remain = None
        self.cur_objective = None
        self.obj_policy = obj_policy

        self.datasets = None  # {ds_a: DatasetWrapper(), ds_b: DatasetWrapper()}
        self.datasets_remain = None
        self.cur_dataset = None

        self.optimizers = None  # {opt_a: Optimizer(), opt_b: Optimizer()}
        self.optimizers_remain = None
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
        self.objectives_remain = list(self.objectives.keys())

        opt_dict = {}
        for cur_opt_name, cur_opt_conf in optimizers_dict.items():
            opt_object = Optimizer(cur_opt_name, cur_opt_conf, obj_dict)
            opt_dict[cur_opt_name] = opt_object
        self.optimizers = opt_dict

        self.relationships = ControllerRelationships(
            optimizers_dict, dataset_dict, objectives_dict
        )

        self._initialize()

        # TODO: validate all connections ds --> obj --> optimizer ?

    def _initialize(self):
        # can use this manually if desired, but is currently dangerous, in that
        # the policy will advance the objective it it is already initialized..
        # this could be changed by setting the current to None if needed
        # TODO: implement `initialize` to allow setting the initial information

        if self.obj_policy:
            first_obj = self.select_objective(cur_policy=self.obj_policy)
        else:
            first_obj = self.objectives_remain[0]
        self.set_by_objective(first_obj)

    def select_objective(self, cur_policy=None):
        # TODO: logging
        policy = cur_policy or self.self.obj_policy

        if not policy:
            # select 'next' as defined
            if not self.cur_objective:
                # initialize to 0 after increment
                cur_ind = -1
            else:
                cur_ind = self.objectives_remain.index(f"{self.cur_objective.name}")
            new_ind = cur_ind + 1
            # wrap
            obj_name = self.objectives_remain[new_ind % len(self.objectives_remain)]
        elif policy == "random":
            # select a ``random'' objective
            new_ind = random.randint(0, len(self.objectives_remain))
            obj_name = self.objectives_remain[new_ind % len(self.objectives_remain)]
        else:
            raise ValueError(
                f"objective policy type {cur_policy} is not currently supported"
            )

        return obj_name

    def maybe_advance_objective(self, obj_policy=None):
        """
        advance the objective according the policy, which if does not exist,
        will advance the ``next'' objective (order of creation)
        """
        if obj_policy:
            cur_policy = obj_policy
        else:
            cur_policy = self.obj_policy

        obj_name = self.select_objective(cur_policy=cur_policy)
        self.set_by_objective(obj_name)

        # return True if advancing
        advanced = self.cur_objective.name == obj_name
        return advanced

    def set_by_objective(self, obj_name):
        self.cur_objective = self.objectives[obj_name]
        self.cur_dataset = self.objectives[obj_name].dataset  # assuming only one?
        self.cur_optimizer = self.objectives[obj_name].optimizer

    # convenience
    def __enter__(self):
        self._initialize()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
