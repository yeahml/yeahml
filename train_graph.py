import tensorflow as tf
import numpy as np
import pickle
from tqdm import tqdm
import os

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
    with open(
        "./experiment/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "wb"
    ) as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# TODO: file name needs to be managed
def load_obj(name):
    with open(
        "./experiment/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "rb"
    ) as f:
        return pickle.load(f)


# TODO: Global needs to be managed, if possible
GLOBAL_SET_TYPE = None
TFR_DIR = "./experiment/cats_v_dogs_01/data/record_holder/150"
BEST_PARAMS_PATH = "best_params"

# TODO: standard dirs need to be created
# make_standard_dirs()


def _parse_function(example_proto):
    global GLOBAL_SET_TYPE
    labelName = str(GLOBAL_SET_TYPE) + "/label"
    featureName = str(GLOBAL_SET_TYPE) + "/image"
    feature = {
        featureName: tf.FixedLenFeature([], tf.string),
        labelName: tf.FixedLenFeature([], tf.int64),
    }

    # decode
    parsed_features = tf.parse_single_example(example_proto, features=feature)

    # convert image data from string to number
    image = tf.decode_raw(parsed_features[featureName], tf.float32)
    # TODO: these values should be acquired from the yaml
    image = tf.reshape(image, [150, 150, 3])
    label = tf.cast(parsed_features[labelName], tf.int64)

    # [do any preprocessing here]

    return image, label


def return_batched_iter(setType, MCd, sess):
    global GLOBAL_SET_TYPE
    global TFR_DIR
    GLOBAL_SET_TYPE = setType

    filenames_ph = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    if GLOBAL_SET_TYPE != "test":
        dataset = dataset.shuffle(buffer_size=MCd["shuffle_buffer"])
    # dataset = dataset.shuffle(buffer_size=1)
    # prefetch is used to ensure one batch is always ready
    # TODO: this prefetch should have some logic based on the
    # system environment, batchsize, and data size
    dataset = dataset.batch(MCd["batch_size"]).prefetch(1)
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    tfrecords_file_name = str(GLOBAL_SET_TYPE) + ".tfrecords"
    tfrecord_file_path = os.path.join(TFR_DIR, tfrecords_file_name)

    # initialize
    sess.run(iterator.initializer, feed_dict={filenames_ph: [tfrecord_file_path]})

    return iterator


def train_graph(g, MCd):
    global BEST_PARAMS_PATH
    saver, init_global, init_local = g.get_collection("save_init")
    X, y_raw, training_op = g.get_collection("main_ops")
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
    root_logs = "./experiment/cats_v_dogs_01"
    train_writer = tf.summary.FileWriter(os.path.join(root_logs, "tf_logs", "train"))
    val_writer = tf.summary.FileWriter(os.path.join(root_logs, "tf_logs", "validation"))

    best_val_loss = np.inf

    with tf.Session(graph=g) as sess:

        # test
        #         test_iter = return_batched_iter('test', MCd, sess)
        #         next_test_element = test_iter.get_next()
        sess.run([init_global, init_local])

        for e in tqdm(range(1, MCd["epochs"] + 1)):
            sess.run(
                [
                    val_met_reset_op,
                    val_loss_reset_op,
                    train_met_reset_op,
                    train_loss_reset_op,
                ]
            )
            # training
            tr_iter = return_batched_iter("train", MCd, sess)
            next_tr_element = tr_iter.get_next()

            # loop entire training set
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
                        feed_dict={X: data, y_raw: target},
                    )
                #                     pr, yt, yp = sess.run([preds, y_true_cls, y_pred_cls], feed_dict={X:data, y_raw:target})
                #                     print(pr)
                #                     print(yt)
                #                     print(yp)
                except tf.errors.OutOfRangeError:
                    break

            # write average for epoch
            summary = sess.run(epoch_train_write_op)
            train_writer.add_summary(summary, e)
            train_writer.flush()

            # run validation
            val_iter = return_batched_iter("val", MCd, sess)
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
