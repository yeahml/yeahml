import tensorflow as tf


class bidir_block(tf.keras.layers.Layer):
    def __init__(self, units=None, return_sequences=False, **kwargs):
        if not units:
            raise ValueError("units are required")
        self.units = units
        self.return_sequences = return_sequences

        super(bidir_block, self).__init__(**kwargs)

        self.rnnlayer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units, return_sequences=return_sequences)
        )

    def get_config(self):
        config = super(bidir_block, self).get_config()
        config.update({"units": self.units})
        config.update({"return_sequences": self.return_sequences})
        return config

    def call(self, inputs):

        out = self.rnnlayer(inputs)

        return out
