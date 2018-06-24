import tensorflow as tf
import pickle
import os
import numpy as np

# Helper to make the output consistent
# TODO: this could sit in a helper file?
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# TODO: this could sit in a helper file?
def load_obj(name):
    with open(
        "./experiment/cats_v_dogs_01/trial/best_params/" + name + ".pkl", "rb"
    ) as f:
        return pickle.load(f)


# TODO: this could sit in a helper file?
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


# TODO: this could sit in a helper file?
def return_batched_iter(setType, MCd, sess):
    global GLOBAL_SET_TYPE
    GLOBAL_SET_TYPE = setType

    filenames_ph = tf.placeholder(tf.string, shape=[None])

    dataset = tf.data.TFRecordDataset(filenames_ph)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    if GLOBAL_SET_TYPE != "test":
        dataset = dataset.shuffle(buffer_size=MCd["shuffle_buffer"])
    # dataset = dataset.shuffle(buffer_size=1)
    dataset = dataset.batch(MCd["batch_size"])
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    tfrecords_file_name = str(GLOBAL_SET_TYPE) + ".tfrecords"
    tfrecord_file_path = os.path.join(MCd["TFR_dir"], tfrecords_file_name)

    # initialize
    sess.run(iterator.initializer, feed_dict={filenames_ph: [tfrecord_file_path]})

    return iterator


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

        test_iter = return_batched_iter("test", MCd, sess)
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
