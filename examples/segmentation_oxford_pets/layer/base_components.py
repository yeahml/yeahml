import tensorflow as tf


class n_by_n(tf.keras.layers.Layer):
    def __init__(
        self,
        kernel_size=None,
        filters=None,
        strides=1,
        padding=None,
        activation=None,
        **kwargs,
    ):
        if not filters:
            raise ValueError("filters are required")
        if not strides:
            raise ValueError("strides are required")
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
            strides=strides,
            padding=padding,
            activation=activation,
            **kwargs,
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
        strides=1,
        padding=None,
        activation=None,
        horizontal_first=True,
        **kwargs,
    ):
        if not filters:
            raise ValueError("filters are required")
        if not strides:
            raise ValueError("strides are required")
        if not padding:
            raise ValueError("padding is required")
        if not kernel_size:
            raise ValueError("kernel_size is required")
        if not activation:
            raise ValueError("activation is required")
        super(n_hv_path, self).__init__(**kwargs)

        self.h_first = horizontal_first

        if self.h_first:
            h_strides = strides
            v_strides = 1
        else:
            h_strides = 1
            v_strides = strides

        # horizontal
        self.gated_h = gated(
            kernel_size=[kernel_size, 1],
            filters=filters,
            strides=h_strides,
            padding=padding,
            activation=activation,
            **kwargs,
        )

        # vertical
        self.gated_v = gated(
            kernel_size=[1, kernel_size],
            filters=filters,
            strides=v_strides,
            padding=padding,
            activation=activation,
            **kwargs,
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


class gated(tf.keras.layers.Layer):
    def __init__(
        self,
        filters=None,
        kernel_size=None,
        strides=1,
        padding=None,
        activation=None,
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
        super(gated, self).__init__(**kwargs)

        # multiply filters by 2 before the split
        self.conv = tf.keras.layers.Conv2D(
            filters=(filters * 2),
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            **kwargs,
        )
        self.activation = tf.keras.layers.Activation(activation)
        self.gate = tf.keras.layers.Activation("sigmoid")
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        o = self.conv(inputs)
        g, z = tf.split(o, 2, axis=-1)
        gate = self.gate(g)
        act = self.activation(z)
        out = self.multiply([gate, act])
        return out
