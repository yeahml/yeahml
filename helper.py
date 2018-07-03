import tensorflow as tf


def print_tensor_info(t):
    print("| {:15s} | {}".format(t.name.rstrip("0123456789").rstrip(":"), t.shape))
