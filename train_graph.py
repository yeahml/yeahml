import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import os

# TODO: make sure global var still works....
from handle_data import return_batched_iter, reinitialize_iter


# these two functions (get_model_params and restore_model_params) are
# ad[a|o]pted from:
# https://github.com/ageron/handson-ml/blob/master/11_deep_learning.ipynb
def get_model_params():
    global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {
        global_vars.op.name: value
        for global_vars, value in zip(
            global_vars, tf.get_default_session().run(global_vars)
        )
    }


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


# these two functions are used to manually save the best
# model params to disk
# TODO: file name needs to be managed
def save_obj(obj, name):
    with open("./example/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# TODO: file name needs to be managed
def load_obj(name):
    with open("./example/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


# TODO: Global needs to be managed, if possible
# GLOBAL_SET_TYPE = None
BEST_PARAMS_PATH = "best_params"

# TODO: standard dirs need to be created
# make_standard_dirs()


def train_graph(g, MCd):
    global BEST_PARAMS_PATH
    saver, init_global, init_local = g.get_collection("save_init")
    X, y_raw, training, training_op = g.get_collection("main_ops")
    preds, y_true_cls, y_pred_cls, _ = g.get_collection("preds")
    train_auc, train_auc_update, train_acc, train_acc_update, train_met_reset_op = g.get_collection(
        "train_metrics"
    )
    val_auc, val_auc_update, val_acc, val_acc_update, val_met_reset_op = g.get_collection(
        "val_metrics"
    )
    train_mean_loss, train_mean_loss_update, train_loss_reset_op = g.get_collection(
        "train_loss"
    )
    val_mean_loss, val_mean_loss_update, val_loss_reset_op = g.get_collection(
        "val_loss"
    )
    epoch_train_write_op, epoch_validation_write_op = g.get_collection("tensorboard")
    #     next_tr_element, next_val_element, _ = g.get_collection("data_sets")

    # TODO: these logs needs to go to the correct place
    root_logs = "./example/cats_v_dogs_01"
    train_writer = tf.summary.FileWriter(os.path.join(root_logs, "tf_logs", "train"))
    val_writer = tf.summary.FileWriter(os.path.join(root_logs, "tf_logs", "validation"))

    best_val_loss = np.inf

    with tf.Session(graph=g) as sess:

        sess.run([init_global, init_local])
        filenames_ph = tf.placeholder(tf.string, shape=[None])
        tr_iter = return_batched_iter("train", MCd, filenames_ph)
        val_iter = return_batched_iter("val", MCd, filenames_ph)

        for e in tqdm(range(1, MCd["epochs"] + 1)):
            sess.run(
                [
                    val_met_reset_op,
                    val_loss_reset_op,
                    train_met_reset_op,
                    train_loss_reset_op,
                ]
            )

            reinitialize_iter(sess, tr_iter, "train", filenames_ph)
            next_tr_element = tr_iter.get_next()

            # loop entire training set
            # main training loop
            while True:
                try:
                    data, target = sess.run(next_tr_element)
                    target = np.reshape(target, (target.shape[0], 1))
                    sess.run(
                        [
                            training_op,
                            train_auc_update,
                            train_acc_update,
                            train_mean_loss_update,
                        ],
                        feed_dict={X: data, y_raw: target, training: True},
                    )
                except tf.errors.OutOfRangeError:
                    break

            # write average for epoch
            summary = sess.run(epoch_train_write_op)
            train_writer.add_summary(summary, e)
            train_writer.flush()

            # run validation
            reinitialize_iter(sess, val_iter, "val", filenames_ph)
            next_val_element = val_iter.get_next()
            while True:
                try:
                    Xb, yb = sess.run(next_val_element)
                    yb = np.reshape(yb, (yb.shape[0], 1))
                    sess.run(
                        [val_auc_update, val_acc_update, val_mean_loss_update],
                        feed_dict={X: Xb, y_raw: yb},
                    )
                except tf.errors.OutOfRangeError:
                    break

            # check for (and save) best validation params here
            # TODO: there should be a flag here as desired
            cur_loss, cur_acc = sess.run([val_mean_loss, val_acc])
            if cur_loss < best_val_loss:
                best_val_loss = cur_loss
                best_params = get_model_params()
                save_obj(best_params, BEST_PARAMS_PATH)
                print(
                    "best params saved: val acc: {:.3f}% val loss: {:.4f}".format(
                        cur_acc * 100, cur_loss
                    )
                )

            summary = sess.run(epoch_validation_write_op)
            val_writer.add_summary(summary, e)
            val_writer.flush()

        train_writer.close()
        val_writer.close()
    return sess
