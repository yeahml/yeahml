import tensorflow as tf

from .gated_conv import gated


class n_by_n(tf.keras.layers.Layer):
    def __init__(
        self, kernel_size=None, filters=None, padding=None, activation=None, **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not padding:
            raise ValueError("padding is required")
        if not kernel_size:
            raise ValueError("kernel_size is required")
        if not activation:
            raise ValueError("activation is required")
        super(n_by_n, self).__init__(**kwargs)

        self.gated = gated(
            kernel_size=kernel_size,
            filters=filters,
            padding=padding,
            activation=activation,
        )

    def call(self, inputs):
        conv_out = self.gated(inputs)
        return conv_out


class n_hv_path(tf.keras.layers.Layer):
    def __init__(
        self,
        some_kwarg=None,  # TODO:
        kernel_size=None,
        filters=None,
        padding=None,
        activation=None,
        horizontal_first=True,
        **kwargs,
    ):
        if not filters:
            raise ValueError("filters are required")
        if not padding:
            raise ValueError("padding is required")
        if not kernel_size:
            raise ValueError("kernel_size is required")
        if not activation:
            raise ValueError("activation is required")
        super(n_hv_path, self).__init__(**kwargs)

        self.h_first = horizontal_first

        # horizontal
        self.gated_h = gated(
            kernel_size=[kernel_size, 1],
            filters=filters,
            padding=padding,
            activation=activation,
        )

        # vertical
        self.gated_v = gated(
            kernel_size=[1, kernel_size],
            filters=filters,
            padding=padding,
            activation=activation,
        )

    def get_config(self):
        config = super(n_hv_path, self).get_config()
        config.update({"horizontal_first": self.h_first})
        return config

    def call(self, inputs):

        if self.h_first:
            first_conv = self.gated_h
            second_conv = self.gated_v
        else:
            first_conv = self.gated_v
            second_conv = self.gated_h

        out_first = first_conv(inputs)
        out_second = second_conv(out_first)

        return out_second
