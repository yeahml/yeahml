# TODO: rename this file &| move components out
import tensorflow as tf
import sys


def create_metrics_ops(
    MCd: dict, set_type: str, y_trues, y_preds, y_vals, pred_vals
) -> tuple:

    # y_vals and pred_vals are the int/float output value
    # y_trues and y_preds are the threshold value
    # this function could be changed to only accept the 'vals'
    # and apply the threshold depending on the type of problem

    met_set = MCd["met_set"]

    report_ops_list = []
    update_ops = []
    scope_str = set_type + "_metrics"
    reset_str = set_type + "_mets_reset"

    with tf.name_scope(scope_str) as scope:

        # TODO: Temp. this may need to change location &| logic
        if MCd["metrics_type"] == "segmentation":
            # this argmax may only be useful for
            y_trues = tf.argmax(y_trues, axis=3, output_type=tf.int64)
            y_preds = tf.argmax(y_preds, axis=3, output_type=tf.int64)

        if "fp" in met_set:
            fp, fp_update = tf.metrics.false_positives(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(fp)
            update_ops.append(fp_update)

        if "fn" in met_set:
            fn, fn_update = tf.metrics.false_negatives(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(fn)
            update_ops.append(fn_update)

        if "tp" in met_set:
            tp, tp_update = tf.metrics.true_positives(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(tp)
            update_ops.append(tp_update)

        if "tn" in met_set:
            tn, tn_update = tf.metrics.true_negatives(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(tn)
            update_ops.append(tn_update)

        if "auc" in met_set:
            # TODO: handle case where the values are bool (before this point)
            # > this may require pushing the bool conversion in this fn
            auc, auc_update = tf.metrics.auc(labels=y_vals, predictions=pred_vals)
            report_ops_list.append(auc)
            update_ops.append(auc_update)

        if "accuracy" in met_set:
            acc, acc_update = tf.metrics.accuracy(labels=y_trues, predictions=y_preds)
            report_ops_list.append(acc)
            update_ops.append(acc_update)

        if "precision" in met_set:
            precision, precision_update = tf.metrics.precision(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(precision)
            update_ops.append(precision_update)

        if "recall" in met_set:
            recall, recall_update = tf.metrics.recall(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(recall)
            update_ops.append(recall_update)

        if "rmse" in met_set:
            rmse, rmse_update = tf.metrics.root_mean_squared_error(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(rmse)
            update_ops.append(rmse_update)

        if "mae" in met_set:
            mae, mae_update = tf.metrics.mean_absolute_error(
                labels=y_trues, predictions=y_preds
            )
            report_ops_list.append(mae)
            update_ops.append(mae_update)

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
        if len(report_ops_list) == 1 and len(update_ops) == 1:
            mets_report_group = report_ops_list[0]
            mets_update_group = update_ops[0]
        else:
            mets_report_group = tf.group(report_ops_list)
            mets_update_group = tf.group(update_ops)
        mets_vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES
        )
        mets_reset = tf.variables_initializer(
            mets_vars, name="performance_metrics_reset"
        )

    return (report_ops_list, mets_report_group, mets_update_group, mets_reset)
