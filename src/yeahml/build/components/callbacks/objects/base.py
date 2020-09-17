import itertools


def implemented(method):
    method._is_implemented = False
    return method


def is_implemented(method):
    """Check if a method is decorated with the `default` wrapper."""
    return getattr(method, "_is_implemented", True)


# NOTE: I'm not sure if an additional abstraction is needed "train"/"inference"
# or Fit "True"/"False"
# CB_TRAIN_OPTIONS = ["train", "inference"]
CB_TIMING_OPTIONS = ["pre", "post"]
CB_STATE_OPTIONS = [
    "task",
    "obtain_task",
    "obtain_dataset",
    "dataset_pass",
    "batch",
    "obtain_batch",
    "prediction",
    "loss",
    "calc_gradient",
    "apply_gradient",
    "metric",
]

# this is a terrible name, but I'm not sure what it should be yet
# the idea is to be allowed to create a relationship between the callback and
# the particular "part" of the training cycle you're interested in.
# NOTE: the issue I'm currently trying to think through is how to organize
# callbacks in a multitask scenario. The issue is that some callbacks (LR
# schedule) should be organized by optimizer and some callbacks should be by
# task (e.g. ______), and some callbacks are task/optimizer agnostic (e.g.
# TerminateOnNan)
# `datasets`
# NOTE: `or`
CB_RELATION_KEY = ["global", "optimizer", "objective", "dataset"]


class Callback:
    """ used to build new callbacks 
    levels


    - train/eval* (implemented by child)
        - task
            - obtain_task
            - 
        - obtain_dataset
            # sample_dataset?
            - dataset_pass (epoch? -- not always applicable) 
                - batch
                    - obtain_batch
                    - prediction
                    - performance
                        - loss
                            - calc_gradient* (train specific)
                            - apply_gradient* (train specific)
                        - metric

    control flow
    - pre (immediately)
    - post (immediately)

    """

    def __init__(self, relation_key=None):
        if not relation_key:
            raise ValueError(
                f"no relation_key {relation_key} detected, please select from {CB_RELATION_KEY}"
            )
        if relation_key not in CB_RELATION_KEY:
            raise ValueError(
                f"{relation_key} not allowed, please select from {CB_RELATION_KEY}"
            )
        self.relation_key = relation_key

    # task
    @implemented
    def pre_task(self):
        """[summary]
        """

    @implemented
    def post_task(self):
        """[summary]
        """

    # obtain_task
    @implemented
    def pre_obtain_task(self):
        """[summary]
        """

    @implemented
    def post_obtain_task(self):
        """[summary]
        """

    # obtain_dataset
    @implemented
    def pre_obtain_dataset(self):
        """[summary]
        """

    @implemented
    def post_obtain_dataset(self):
        """[summary]
        """

    # dataset_pass
    @implemented
    def pre_dataset_pass(self):
        """[summary]
        """

    @implemented
    def post_dataset_pass(self):
        """[summary]
        """

    # batch
    @implemented
    def pre_batch(self):
        """[summary]
        """

    @implemented
    def post_batch(self):
        """[summary]
        """

    # obtain_data
    @implemented
    def pre_obtain_batch(self):
        """[summary]
        """

    @implemented
    def post_obtain_batch(self):
        """[summary]
        """

    # prediction
    @implemented
    def pre_prediction(self):
        """[summary]
        """

    @implemented
    def post_prediction(self):
        """[summary]
        """

    # performance
    @implemented
    def pre_performance(self):
        """[summary]
        """

    @implemented
    def post_performance(self):
        """[summary]
        """

    # loss
    @implemented
    def pre_loss(self):
        """[summary]
        """

    @implemented
    def post_loss(self):
        """[summary]
        """

    # metric
    @implemented
    def pre_metric(self):
        """[summary]
        """

    @implemented
    def post_metric(self):
        """[summary]
        """


class TrainCallback(Callback):
    # calc gradient
    @implemented
    def pre_calc_gradient(self):
        """[summary]
        """

    @implemented
    def post_calc_gradient(self):
        """[summary]
        """

    # apply gradient
    @implemented
    def pre_apply_gradient(self):
        """[summary]
        """

    @implemented
    def post_apply_gradient(self):
        """[summary]
        """


class Callbacks:
    def __init__(self, callbacks):
        # TODO: could check to ensure these are valid callbacks
        self.callbacks = callbacks if callbacks else None

        # create dictionary of the callbacks to call at the specified time+state
        cb_dict = {}
        if self.callbacks:
            combos = itertools.product(CB_TIMING_OPTIONS, CB_STATE_OPTIONS)
            for cb in self.callbacks:
                for cb_name_tup in combos:
                    # e.g. pre_task
                    cb_name = f"{cb_name_tup[0]}_{cb_name_tup[1]}"
                    try:
                        _ = cb_dict[cb_name]
                    except KeyError:
                        cb_dict[cb_name] = []
                    cb_method = getattr(cb, cb_name)
                    if is_implemented(cb_method):
                        cb_dict[cb_name].append(cb_method)
        self.cb_dict = cb_dict

    def pre_task(self):
        if self.cb_dict["pre_task"]:
            for cb_method in self.cb_dict["pre_task"]:
                cb_method()

    # def pre_task(self):
    #     if self.cb_dict["pre_task"]:
    #         for cb_method in self.cb_dict["pre_task"]:
    #             cb_method()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class CallbackContainer:
    def __init__(
        self, callbacks, optimizer_names=None, objective_names=None, dataset_names=None
    ):
        self.callbacks = callbacks if callbacks else None
        if not optimizer_names:
            raise ValueError("no optimizers detected")
        if not objective_names:
            raise ValueError("no objectives detected")
        if not dataset_names:
            raise ValueError("no datasets detected")

        # TODO: in these two preceding blocks; there is likely a more elegant
        # way to approach this
        opt_, obj_, ds_, g_ = [], [], [], []
        for cb in self.callbacks:
            if cb.relation_key == "global":
                g_.append(cb)
            elif cb.relation_key == "optimizer":
                opt_.append(cb)
            elif cb.relation_key == "objective":
                obj_.append(cb)
            elif cb.relation_key == "dataset":
                ds_.append(cb)
        rel_dict = {}
        for rel_def in CB_RELATION_KEY:
            if rel_def == "global":
                rel_dict[rel_def] = Callbacks(g_)
            elif rel_def == "optimizer":
                # TODO: loop names
                rel_dict[rel_def] = Callbacks(opt_)
            elif rel_def == "objective":
                # TODO: loop names
                rel_dict[rel_def] = Callbacks(obj_)
            elif rel_def == "dataset":
                # TODO: loop names
                rel_dict[rel_def] = Callbacks(ds_)
        self.rel_dict = rel_dict

    def pre_task(self, rel_key="global", rel_name=None):
        # TODO: I don't like this... (the rel_key/rel_name and resulting code
        # blocks...)
        if rel_key == "global":
            if self.rel_dict:
                if self.rel_dict[rel_key]:
                    self.rel_dict[rel_key].pre_task()

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
