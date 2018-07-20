# TODO: rename this file &| move components out
import tensorflow as tf


def build_mets_write_op(met_ops: list, loss_op, set_type: str):
    scalars = []
    for t in met_ops:
        name_str = t.name.split("/")[-2]
        if name_str == set_type + "_metrics":
            # single metric case
            name_str = t.name.split("/")[-1]
        tmp_str = name_str + "/" + set_type
        temp_scalar = tf.summary.scalar(tmp_str, t)
        scalars.append(temp_scalar)
    scalar_name = "loss/" + set_type
    loss_scalar = tf.summary.scalar(scalar_name, loss_op)
    scalars.append(loss_scalar)
    write_op_name = set_type + "_metrics_write_op"
    write_op = tf.summary.merge(scalars, name=write_op_name)
    return write_op


def build_loss_ops(batch_loss, set_type: str) -> tuple:
    scope_str = set_type + "_loss_eval"
    reset_str = set_type + "_loss_reset_op"
    with tf.name_scope(scope_str) as scope:
        mean_loss, mean_loss_update = tf.metrics.mean(batch_loss)
        loss_vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        loss_reset = tf.variables_initializer(loss_vars, name=reset_str)
    return (mean_loss, mean_loss_update, loss_reset)

