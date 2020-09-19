from yeahml.build.components.callbacks.objects.base import TrainCallback


def print_mapper(cur_func):
    def _print_mapper(self, *args, **kwargs):
        print(f"{cur_func.__name__}: {self.monitor}")

    return _print_mapper


class Printer(TrainCallback):
    def __init__(self, monitor="something", relation_key=None):
        super(Printer, self).__init__(relation_key=relation_key)
        self.monitor = monitor

    # def pre_task(self):
    #     """[summary]
    #     """
    #     print(f"pre_task: {self.monitor}")

    # def post_task(self):
    #     """[summary]
    #     """
    #     print(f"post_task: {self.monitor}")

    # def pre_metric(self):
    #     """[summary]
    #     """
    #     print(f"pre_metric: {self.monitor}")

    # task
    @print_mapper
    def pre_task(self):
        """[summary]
        """

    @print_mapper
    def post_task(self):
        """[summary]
        """

    @print_mapper
    def pre_obtain_task(self):
        """[summary]
        """

    @print_mapper
    def post_obtain_task(self):
        """[summary]
        """

    @print_mapper
    # obtain_dataset
    def pre_obtain_dataset(self):
        """[summary]
        """

    @print_mapper
    def post_obtain_dataset(self):
        """[summary]
        """

    @print_mapper
    # dataset_pass
    def pre_dataset_pass(self):
        """[summary]
        """

    @print_mapper
    def post_dataset_pass(self):
        """[summary]
        """

    # batch
    @print_mapper
    def pre_batch(self):
        """[summary]
        """

    @print_mapper
    def post_batch(self):
        """[summary]
        """

    # obtain_data
    @print_mapper
    def pre_obtain_batch(self):
        """[summary]
        """

    @print_mapper
    def post_obtain_batch(self):
        """[summary]
        """

    # prediction
    @print_mapper
    def pre_prediction(self):
        """[summary]
        """

    @print_mapper
    def post_prediction(self):
        """[summary]
        """

    # performance
    @print_mapper
    def pre_performance(self):
        """[summary]
        """

    @print_mapper
    def post_performance(self):
        """[summary]
        """

    # loss
    @print_mapper
    def pre_loss(self):
        """[summary]
        """

    @print_mapper
    def post_loss(self):
        """[summary]
        """

    @print_mapper
    # metric
    def pre_metric(self):
        """[summary]
        """

    @print_mapper
    def post_metric(self):
        """[summary]
        """

    @print_mapper
    def pre_calc_gradient(self):
        """[summary]
        """

    @print_mapper
    def post_calc_gradient(self):
        """[summary]
        """

    # apply gradient
    @print_mapper
    def pre_apply_gradient(self):
        """[summary]
        """

    @print_mapper
    def post_apply_gradient(self):
        """[summary]
        """
