from yeahml.build.components.callbacks.objects.base import TrainCallback


class Printer(TrainCallback):
    def __init__(self, monitor="something"):
        super(Printer, self).__init__()
        self.monitor = monitor

    def pre_task(self):
        """[summary]
        """
        print(f"on_batch_begin: {self.monitor}")
