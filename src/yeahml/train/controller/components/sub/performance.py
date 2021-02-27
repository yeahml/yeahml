# from yeahml.build.components.loss import configure_loss


class Performance:
    # NOTE: what if we want to keep track of metrics that don't rely on the
    # current head/output graph? (would need an outter metric)
    def __init__(self, loss_config, metric_config):
        self.loss = loss_config
        self.metric_dict = metric_config

        # losses and metrics are different in tf. that is losses don't have
        # internal state, whereas metrics do

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
