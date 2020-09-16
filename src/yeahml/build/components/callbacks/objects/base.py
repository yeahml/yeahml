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

    def __init__(self):
        pass

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

