import tensorflow as tf

COMPONENT_DICT = {
    "layers": {
        "type": {
            "name": "class",
            "options": {"class": tf.keras.layers, "subclass": tf.keras.layers.Layer},
        }
    },
    "activations": {
        "type": {
            "name": "callable",
            "options": {
                "callable": tf.keras.activations,
                "exclude_list": ["serialize", "deserialize", "get"],
            },
        }
    },
    "optimizers": {
        "type": {"name": "class", "options": {"class": tf.keras.optimizers}}
    },
}

