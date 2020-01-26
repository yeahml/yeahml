import tensorflow as tf
import inspect

"""
# callable template
"_____": {
    "type": {
        "name": "callable",
        "options": {
            "callable": _____,
            "exclude_list": [_____],
        },
    }
},


# class template
"_____": {
    "type": {
        "name": "class",
        "options": {
            "class": _____,
            "subclass": _____,
            "exclude_list": [_____],
        },
    }
}
"""


COMPONENT_DICT = {
    "layer": {
        "type": {
            "name": "class",
            "options": {"class": tf.keras.layers, "subclass": tf.keras.layers.Layer},
        }
    },
    "activation": {
        "type": {
            "name": "callable",
            "options": {
                "callable": tf.keras.activations,
                "exclude_list": ["serialize", "deserialize", "get"],
            },
        }
    },
    "optimizer": {
        "type": {
            "name": "class",
            "options": {
                "class": tf.keras.optimizers,
                "exclude_list": ["serialize", "deserialize", "get", "schedules"],
            },
        }
    },
    "constraint": {
        "type": {
            "name": "class",
            "options": {
                "class": tf.keras.constraints,
                "subclass": tf.keras.constraints.Constraint,
                "exclude_list": ["serialize", "deserialize", "get"],
            },
        }
    },
    "dtype": {
        "type": {
            "name": "class",
            "options": {
                "class": tf.dtypes,
                "subclass": tf.dtypes.DType,
                "exclude_list": ["deserialize", "get", "serialize"],
            },
        }
    },
    "initializer": {
        "type": {
            "name": "class",
            "options": {
                "class": tf.keras.initializers,
                "subclass": tf.keras.initializers.Initializer,
                "exclude_list": ["deserialize", "get", "serialize"],
            },
        }
    },
    "loss": {
        "type": {
            "name": "class",
            "options": {
                "class": tf.losses,
                "exclude_list": ["deserialize", "get", "serialize"],
            },
        }
    },
    "regularizer": {
        "type": {
            "name": "callable",
            "options": {
                "callable": tf.keras.regularizers,
                "exclude_list": ["deserialize", "get", "serialize"],
            },
        }
    },
}
