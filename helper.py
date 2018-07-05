import tensorflow as tf


def fmt_tensor_info(t):
    return "| {:15s} | {}".format(t.name.rstrip("0123456789").rstrip(":"), t.shape)
