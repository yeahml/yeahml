from yeahml.train.controller.components.sub.performance import Performance


class Objective:
    def __init__(self, name, config, ds_dict):

        # NOTE: there may not be an optimizer here if this is only a metric?
        self.optimizer = None  # Optimizer()
        self.name = name

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
                f"optimizer previously set as {self.optimizer}, cannot overwrite to {opt_object}"
            )

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
