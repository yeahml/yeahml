import tensorflow as tf
import os
import numpy as np

from yamlflow.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yamlflow.log.yf_logging import config_logger  # custom logging
from yamlflow.helper import fmt_metric_summary


# TODO: this will need to be updated to match ..from_saver.
# TODO: important components from these two functions may be able to be merged
# to avoid inconsistencies
def eval_graph(g, MCd):
    return


#     logger = config_logger(MCd, "eval")
#     with tf.Session(graph=g) as sess:
#         saver = tf.train.Saver()
#         X, y_raw, training, training_op = g.get_collection("main_ops")
#         preds, y_true_cls, y_pred_cls, _ = g.get_collection("preds")
#         test_auc, test_auc_update, test_acc, test_acc_update, test_acc_reset_op = g.get_collection(
#             "test_metrics"
#         )
#         test_mean_loss, test_mean_loss_update, test_loss_reset_op = g.get_collection(
#             "test_loss"
#         )

#         # restore_model_params(model_params=best_params, g=g, sess=sess)
#         saver.restore(sess, MCd["saver_save"])
#         sess.run([test_acc_reset_op, test_loss_reset_op])

#         filenames_ph = tf.placeholder(tf.string, shape=[None])
#         test_iter = return_batched_iter("test", MCd, filenames_ph)

#         tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_test"])
#         sess.run(test_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})

#         next_test_element = test_iter.get_next()
#         while True:
#             try:
#                 Xb, yb, ib = sess.run(next_test_element)
#                 yb = np.reshape(yb, (yb.shape[0], 1))
#                 sess.run(
#                     [test_auc_update, test_acc_update, test_mean_loss_update],
#                     feed_dict={X: Xb, y_raw: yb},
#                 )
#                 xpp, xgt, xpc = sess.run(
#                     [preds, y_true_cls, y_pred_cls], feed_dict={X: Xb, y_raw: yb}
#                 )
#                 # xiid = sess.run(inst_id_ph, feed_dict={y_raw: ib})
#                 print(ib)
#             except tf.errors.OutOfRangeError:
#                 break

#         # print
#         final_test_acc, final_test_loss, final_test_auc = sess.run(
#             [test_acc, test_mean_loss, test_auc]
#         )
#         print(
#             "test auc: {:.3f}% acc: {:.3f}% loss: {:.5f}".format(
#                 final_test_auc * 100, final_test_acc * 100, final_test_loss
#             )
#         )


def eval_graph_from_saver(MCd):
    logger = config_logger(MCd, "eval")
    logger.debug("eval_graph_from_saver")
    preds_logger = config_logger(MCd, "preds")

    with tf.Session() as sess:
        graph_path = os.path.join(MCd["saver_save"] + ".meta")
        saver = tf.train.import_meta_graph(graph_path)
        g = tf.get_default_graph()
        X, y_raw, training, training_op = g.get_collection("main_ops")
        preds = g.get_collection("preds")
        y_true, y_pred = g.get_collection("gt_and_pred")
        test_mets_report, test_mets_update, test_mets_reset = g.get_collection(
            "test_metrics"
        )
        test_mean_loss, test_mean_loss_update, test_loss_reset_op = g.get_collection(
            "test_loss"
        )
        # index is used here to return the node (not a list of the node)
        epoch_test_write_op = g.get_collection("tensorboard_test")[0]

        # restore_model_params(model_params=best_params, g=g, sess=sess)
        saver.restore(sess, MCd["saver_save"])
        sess.run([test_mets_reset, test_loss_reset_op])

        filenames_ph = tf.placeholder(tf.string, shape=[None])
        test_iter = return_batched_iter("test", MCd, filenames_ph)

        tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_test"])
        sess.run(test_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})

        next_test_element = test_iter.get_next()
        # TODO: temp_ind needs to be replaced with the real index
        temp_ind = 0
        while True:
            temp_ind += 1
            try:
                # Xb, yb, ib = sess.run(next_test_element)
                Xb, yb = sess.run(next_test_element)

                sess.run(
                    [test_mets_update, test_mean_loss_update],
                    feed_dict={X: Xb, y_raw: yb},
                )
                y_gt, y_p = sess.run([y_true, y_pred], feed_dict={X: Xb, y_raw: yb})
                for i, _ in enumerate(Xb):
                    # TODO: determine sane way to include v.. maybe if number features < n?
                    preds_logger.info(
                        "{} gt: {} p: {}".format(temp_ind + i, y_gt[i], y_p[i])
                    )

            except tf.errors.OutOfRangeError:
                break

        summary = sess.run(epoch_test_write_op)
        summary_dict = fmt_metric_summary(summary)
        logger.info("Test metrics: {}".format(summary_dict))
        print(summary_dict)  # can also increase stdout log level
