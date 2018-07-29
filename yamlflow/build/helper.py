# TODO: rename this file &| move components out
import tensorflow as tf
import sys


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


def create_metrics_ops(MCd: dict, set_type: str, y_trues, y_preds) -> tuple:

    met_set = MCd["met_set"]

    report_ops_list = []
    update_ops = []
    scope_str = set_type + "_metrics"
    reset_str = set_type + "_mets_reset"

    with tf.name_scope(scope_str) as scope:
        if "auc" in met_set:
            train_auc, train_auc_update = tf.metrics.auc(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_auc)
            update_ops.append(train_auc_update)

        if "accuracy" in met_set:
            train_acc, train_acc_update = tf.metrics.accuracy(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_acc)
            update_ops.append(train_acc_update)

        if "precision" in met_set:
            train_precision, train_precision_update = tf.metrics.precision(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_precision)
            update_ops.append(train_precision_update)

        if "recall" in met_set:
            train_recall, train_recall_update = tf.metrics.recall(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_recall)
            update_ops.append(train_recall_update)

        if "rmse" in met_set:
            train_rmse, train_rmse_update = tf.metrics.root_mean_squared_error(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_rmse)
            update_ops.append(train_rmse_update)

        if "mae" in met_set:
            train_mae, train_mae_update = tf.metrics.mean_absolute_error(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(train_mae)
            update_ops.append(train_mae_update)

        if "iou" in met_set:
            mean_iou, mean_iou_update = tf.metrics.mean_iou(
                labels=y_trues,
                predictions=y_preds,
                num_classes=MCd["num_classes"],
                weights=None,
            )
            report_ops_list.append(mean_iou)
            update_ops.append(mean_iou_update)

        # Group metrics
        mets_report_group = tf.group(report_ops_list)
        mets_update_group = tf.group(update_ops)
        mets_vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        mets_reset = tf.variables_initializer(mets_vars, name="train_mets_reset")

    return (report_ops_list, mets_report_group, mets_update_group, mets_reset)
