from tensorflow.keras.callbacks import Callback


class Printer(Callback):
    def __init__(self, monitor="something"):
        super(Printer, self).__init__()
        self.monitor = monitor

    # NOTE: I don't understand why both params (self and logs) are necessary..

    # batch
    def on_batch_begin(self, logs=None):
        print(f"on_batch_begin: {self.monitor}")

    def on_batch_end(self, logs=None):
        print(f"on_batch_end: {self.monitor}")

    # train
    def on_train_begin(self, logs=None):
        print(f"on_train_begin: {self.monitor}")

    def on_train_end(self, logs=None):
        print(f"on_train_end: {self.monitor}")

    # epoch
    def on_epoch_begin(self, epoch_num, logs=None):
        print(f"on_epoch_begin: {self.monitor} {epoch_num}")

    def on_epoch_end(self, logs=None):
        print(f"on_epoch_end: {self.monitor}")

    # additional
    # def set_model(self):
    #     print(f"set_model: {self.monitor}")

    # def set_params(self):
    #     print(f"set_params: {self.monitor}")

    # train
    def on_train_begin(self, logs=None):
        print(f"on_train_begin: {self.monitor}")

    def on_train_end(self, logs=None):
        print(f"on_train_end: {self.monitor}")

    # train_batch
    def on_train_batch_begin(self, logs=None):
        print(f"on_train_batch_begin: {self.monitor}")

    def on_train_batch_end(self, logs=None):
        print(f"on_train_batch_end: {self.monitor}")

    # test
    def on_test_begin(self, logs=None):
        print(f"on_test_begin: {self.monitor}")

    def on_test_end(self, logs=None):
        print(f"on_test_end: {self.monitor}")

    # predict
    def on_predict_end(self, logs=None):
        print(f"on_predict_end: {self.monitor}")

    def on_predict_begin(self, logs=None):
        print(f"on_predict_begin: {self.monitor}")

    # predict_batch
    def on_predict_batch_begin(self, logs=None):
        print(f"on_predict_batch_begin: {self.monitor}")

    def on_predict_batch_end(self, logs=None):
        print(f"on_predict_batch_end: {self.monitor}")

