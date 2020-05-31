import tensorflow as tf

from .base_components import n_by_n
from .block_components import multipath, multipath_reduction, upsample


class up_block(tf.keras.layers.Layer):
    def __init__(
        self, filters=None, upsize=2, padding="same", activation=None, **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not upsize:
            raise ValueError("upsize is required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        super(up_block, self).__init__(**kwargs)
        self.upsize = upsize

        self.up = upsample(
            filters=filters,
            upsize=upsize,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.conv = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )

    def get_config(self):
        config = super(up_block, self).get_config()
        config.update({"upsize": self.upsize})
        return config

    def call(self, inputs):

        upsampled = self.up(inputs)
        out = self.conv(upsampled)

        return out


class down_block(tf.keras.layers.Layer):
    def __init__(
        self, filters=None, down_size=None, padding="same", activation=None, **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not down_size:
            raise ValueError("down_size is required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        self.down_size = down_size
        super(down_block, self).__init__(**kwargs)
        self.conv_a = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )
        self.conv_b = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )
        self.concat = tf.keras.layers.Concatenate()
        self.conv_1x1 = n_by_n(
            kernel_size=1,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.down = multipath_reduction(
            filters=filters,
            strides=down_size,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.conv_out = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )

    def get_config(self):
        config = super(up_block, self).get_config()
        config.update({"down_size": self.down_size})
        return config

    def call(self, inputs):

        out_a = self.conv_a(inputs)
        out_a = self.conv_b(out_a)
        out_b = self.concat([out_a, inputs])
        out = self.conv_1x1(out_b)

        out = self.down(out)
        out = self.conv_out(out)

        return out
