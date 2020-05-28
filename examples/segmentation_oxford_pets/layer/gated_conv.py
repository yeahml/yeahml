import tensorflow as tf


class gated(tf.keras.layers.Layer):
    def __init__(
        self,
        some_kwarg=None,  # TODO:
        filters=None,
        kernel_size=None,
        padding=None,
        activation=None,
        **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not padding:
            raise ValueError("padding is required")
        if not kernel_size:
            raise ValueError("kernel_size is required")
        if not activation:
            raise ValueError("activation is required")
        super(gated, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding, activation=None
        )
        self.activation = tf.keras.layers.Activation(activation)
        self.gate = tf.keras.layers.Activation("sigmoid")
        self.multiply = tf.keras.layers.Multiply()

    def get_config(self):
        config = super(gated, self).get_config()
        config.update({"jack": self.jack})
        return config

    def call(self, inputs):
        o = self.conv(inputs)
        g, z = tf.split(o, 2, axis=-1)
        gate = self.gate(g)
        act = self.activation(z)
        out = self.multiply([gate, act])
        return out
