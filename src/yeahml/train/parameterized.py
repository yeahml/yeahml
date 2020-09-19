"""

Attempting to codify the relationships that could exist

left to right: datasets -->  objectives -->  optimizers

that is, objectives use multiple datasets, but are optimized by a single
optimizer. Optimizers can optimize multiple objectives. objectives can learn
from multiple datasets. Each ojective has a performance object with at least 1
loss and N metrics

"""

######################################
######################################


class Optimizer:
    def __init__(self):
        self.objectives = None  # [Objective(), Objective()]
        # self.get_grads_fn = None
        # self.apply_grads_fn = None


######################################


class Performance:
    # TODO: supervised vs unsupervised
    def __init__(self):
        self.loss = None  # single loss
        # NOTE: what if we want to keep track of metrics that don't rely on the
        # current head/output graph? (would need an outter metric)
        self.metrics = None  # [metric_a, metric_b]
        # self.target = None
        # self.output = None


class Objective:
    def __init__(self):
        self.performance = None  # Performance()
        self.optimizer = None  # Optimizer()
        self.datasets = None  # [Dataset(), Dataset()]


######################################


class Dataset:
    def __init__(self):
        self.objectives = None  # [Objective(), Objective()]


######################################
######################################

# TODO: validate connections?
