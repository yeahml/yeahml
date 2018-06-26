import tensorflow as tf
import pickle
import os
import numpy as np

from handle_data import return_batched_iter

# Helper to make the output consistent
# TODO: this could sit in a helper file?
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# TODO: this could sit in a helper file?
def load_obj(name):
    with open("./example/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


# TODO: this could sit in a helper file?
def restore_model_params(model_params, g, sess):
    gvar_names = list(model_params.keys())
    assign_ops = {
        gvar_name: g.get_operation_by_name(gvar_name + "/Assign")
        for gvar_name in gvar_names
    }
    init_values = {
        gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()
    }
    feed_dict = {
        init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names
    }
    sess.run(assign_ops, feed_dict=feed_dict)


def eval_graph(g, MCd):

    best_params = load_obj(MCd["save_pparams"])
    with tf.Session(graph=g) as sess:
        saver, init_global, init_local = g.get_collection("save_init")
        X, y_raw, training, training_op = g.get_collection("main_ops")
        preds, y_true_cls, y_pred_cls, _ = g.get_collection("preds")
        test_auc, test_auc_update, test_acc, test_acc_update, test_acc_reset_op = g.get_collection(
            "test_metrics"
        )
        test_mean_loss, test_mean_loss_update, test_loss_reset_op = g.get_collection(
            "test_loss"
        )

        restore_model_params(model_params=best_params, g=g, sess=sess)
        sess.run([test_acc_reset_op, test_loss_reset_op])

        filenames_ph = tf.placeholder(tf.string, shape=[None])
        test_iter = return_batched_iter("test", MCd, filenames_ph)

        tfr_f_path = os.path.join(MCd["TFR_dir"], "test.tfrecords")
        sess.run(test_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})

        next_test_element = test_iter.get_next()
        while True:
            try:
                Xb, yb = sess.run(next_test_element)
                yb = np.reshape(yb, (yb.shape[0], 1))
                sess.run(
                    [test_auc_update, test_acc_update, test_mean_loss_update],
                    feed_dict={X: Xb, y_raw: yb},
                )
            except tf.errors.OutOfRangeError:
                break

        # print
        final_test_acc, final_test_loss, final_test_auc = sess.run(
            [test_acc, test_mean_loss, test_auc]
        )
        print(
            "test auc: {:.3f}% acc: {:.3f}% loss: {:.5f}".format(
                final_test_auc * 100, final_test_acc * 100, final_test_loss
            )
        )
