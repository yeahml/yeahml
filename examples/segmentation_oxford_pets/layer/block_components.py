import tensorflow as tf

from .base_components import n_by_n, n_hv_path


class multipath(tf.keras.layers.Layer):
    def __init__(self, filters=None, padding=None, activation=None, **kwargs):
        if not filters:
            raise ValueError("filters are required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        super(multipath, self).__init__(**kwargs)

        ######## path a
        # subpath a
        self.path_a_a_1 = n_by_n(
            kernel_size=3,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.path_a_a_2 = n_by_n(
            kernel_size=3,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

        # subpath b
        self.path_a_b = n_by_n(
            kernel_size=3,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

        self.path_a_add = tf.keras.layers.Add()

        ######## path b
        # subpath c
        self.path_b_a = n_hv_path(
            kernel_size=5,
            filters=filters,
            padding=padding,
            activation=activation,
            horizontal_first=True,
            **kwargs
        )

        # subpath d
        self.path_b_b = n_hv_path(
            kernel_size=7,
            filters=filters,
            padding=padding,
            activation=activation,
            horizontal_first=False,
            **kwargs
        )
        self.path_b_add = tf.keras.layers.Add()

        ##### out
        self.out_concat = tf.keras.layers.Concatenate()
        self.path_out_1x1 = n_by_n(
            kernel_size=1,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

    def call(self, inputs):

        ####### path a
        path_a_a_1 = self.path_a_a_1(inputs)
        path_a_a_out = self.path_a_a_2(path_a_a_1)
        path_a_b_out = self.path_a_b(inputs)
        out_a = self.path_a_add([path_a_a_out, path_a_b_out])

        ####### path b
        path_b_a_out = self.path_b_a(inputs)
        path_b_b_out = self.path_b_b(inputs)
        out_b = self.path_b_add([path_b_a_out, path_b_b_out])

        #### out
        concat_ab = self.out_concat([out_a, out_b])
        out = self.path_out_1x1(concat_ab)

        return out


class upsample(tf.keras.layers.Layer):
    def __init__(self, filters=None, upsize=2, padding=None, activation=None, **kwargs):
        if not filters:
            raise ValueError("filters are required")
        if not upsize:
            raise ValueError("upsize is required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        super(upsample, self).__init__(**kwargs)

        self.nearest = tf.keras.layers.UpSampling2D(
            size=(upsize, upsize), interpolation="nearest", **kwargs
        )
        self.bilinear = tf.keras.layers.UpSampling2D(
            size=(upsize, upsize), interpolation="bilinear", **kwargs
        )
        self.deconv = tf.keras.layers.Conv2DTranspose(
            kernel_size=3,
            filters=filters,
            strides=upsize,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.concat = tf.keras.layers.Concatenate()
        self.conv1x1_feat = n_by_n(
            kernel_size=1,
            filters=(filters * 2),
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.conv1x1_out = n_by_n(
            kernel_size=1,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

    def get_config(self):
        config = super(upsample, self).get_config()
        config.update({"upsize": self.upsize})
        return config

    def call(self, inputs):
        a = self.nearest(inputs)
        b = self.bilinear(inputs)
        c = self.deconv(inputs)
        conc = self.concat([a, b, c])
        feat = self.conv1x1_feat(conc)
        out = self.conv1x1_out(feat)
        return out


class multipath_reduction(tf.keras.layers.Layer):
    def __init__(
        self, filters=None, strides=None, padding=None, activation=None, **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not strides:
            raise ValueError("strides is required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        super(multipath_reduction, self).__init__(**kwargs)

        ######## path a
        # subpath a
        self.path_a_a_1 = n_by_n(
            kernel_size=3,
            filters=filters,
            strides=strides,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.path_a_a_2 = n_by_n(
            kernel_size=3,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

        # subpath b
        self.path_a_b = n_by_n(
            kernel_size=3,
            filters=filters,
            padding=padding,
            strides=strides,
            activation=activation,
            **kwargs
        )

        self.path_a_add = tf.keras.layers.Add()

        ######## path b
        # subpath c
        self.path_b_a = n_hv_path(
            kernel_size=3,
            filters=filters,
            padding=padding,
            strides=strides,
            activation=activation,
            horizontal_first=True,
            **kwargs
        )

        # subpath d
        self.path_b_b = n_hv_path(
            kernel_size=3,
            filters=filters,
            strides=strides,
            padding=padding,
            activation=activation,
            horizontal_first=False,
            **kwargs
        )
        self.path_b_add = tf.keras.layers.Add()

        ##### out
        self.out_concat = tf.keras.layers.Concatenate()
        self.path_out_1x1 = n_by_n(
            kernel_size=1,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )

    def call(self, inputs):

        ####### path a
        path_a_a_1 = self.path_a_a_1(inputs)
        path_a_a_out = self.path_a_a_2(path_a_a_1)
        path_a_b_out = self.path_a_b(inputs)
        out_a = self.path_a_add([path_a_a_out, path_a_b_out])

        ####### path b
        path_b_a_out = self.path_b_a(inputs)
        path_b_b_out = self.path_b_b(inputs)
        out_b = self.path_b_add([path_b_a_out, path_b_b_out])

        #### out
        concat_ab = self.out_concat([out_a, out_b])
        out = self.path_out_1x1(concat_ab)

        return out
