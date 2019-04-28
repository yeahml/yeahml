import tensorflow as tf


def get_regularizer_fn(reg_ops: dict):

    if not reg_ops:
        return None

    try:
        reg_type = reg_ops["type"].lower()
        if reg_type not in ["l1", "l2", "l1l2"]:
            raise NotImplementedError(
                f"reg_type {reg_type} not implemented, allowed: [l1,l2,l1l2]"
            )
    except KeyError:
        raise ValueError("reg_type not specified, allowed: [l1,l2,l1l2]")

    try:
        scale = reg_ops["scale"]
        # TODO: ensure scale is a number (within range preferably)
        # TODO: handle list/value
    except KeyError:
        raise ValueError("scale not specified")

    if reg_type == "l1":
        # TODO: handle list/value
        reg_fn = tf.keras.regularizers.l1(l=scale[0])
    elif reg_type == "l2":
        # TODO: handle list/value
        reg_fn = tf.keras.regularizers.l2(l=scale[0])
    elif reg_type == "l1l2":
        # TODO: handle list/value and multiple values
        reg_fn = tf.keras.regularizers.L1L2(l1=scale[0], l2=scale[1])
    else:
        raise NotImplementedError(
            f"regularization type {reg_str} not implemented, only [l1,l2,l1l2]"
        )

    return reg_fn
