import tensorflow as tf
import math
from tqdm import tqdm
import os
import sys

# import custom logging
from yf_logging import config_logger


# TODO: make sure global var still works....
from handle_data import return_batched_iter

# from helper import load_obj, save_obj, get_model_params

# used for loading params
from load_params_onto_layer import init_params_from_file


def train_graph(g, MCd: dict, HCd: dict):
    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    EARLY_STOPPING_e = MCd["early_stopping_e"]  # default is preset to 0
    WARM_UP_e = MCd["warm_up_epochs"]  # default is 3

    init_global, init_local = g.get_collection("init")
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
    epoch_train_write_op, epoch_validation_write_op, hist_write_op = g.get_collection(
        "tensorboard"
    )

    best_val_loss = math.inf
    last_best_e = 0  # marker for early stopping

    with tf.Session(graph=g) as sess:

        train_writer = tf.summary.FileWriter(
            os.path.join(MCd["log_dir"], "tf_logs", "train"), graph=sess.graph
        )
        val_writer = tf.summary.FileWriter(
            os.path.join(MCd["log_dir"], "tf_logs", "validation")
        )

        sess.run([init_global, init_local])
        saver = tf.train.Saver()  # create after initializing variables

        ## TL, INIT_DEBUG, pre loading
        # print("TRANSFER INIT : {}".format(MCd["load_params_path"]))
        # AA, BB = 1, 0
        # dense_1_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dense_1")
        # dense_1_vars = [
        #     v
        #     for v in dense_1_vars
        #     if v.name.rstrip("0123456789").endswith("kernel:")
        #     or v.name.rstrip("0123456789").endswith("bias:")
        # ]
        # # print(dense_1_vars)
        # d1_pre = sess.run(dense_1_vars)
        # print(d1_pre[AA][BB])

        # initialize variables from file as specified (as used in transfer learning)
        # TODO: I would rather this have a successful return, rather than act as a method
        init_params_from_file(sess, MCd, HCd)

        # TL, INIT_DEBUG, post loading
        # d1_post = sess.run(dense_1_vars)
        # print(d1_post[AA][BB])

        filenames_ph = tf.placeholder(tf.string, shape=[None])
        tr_iter = return_batched_iter("train", MCd, filenames_ph)
        val_iter = return_batched_iter("val", MCd, filenames_ph)

        # tracing options
        try:
            if MCd["trace_level"].lower() == "full":
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            elif MCd["trace_level"].lower() == "software":
                run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
            elif MCd["trace_level"].lower() == "hardware":
                run_options = tf.RunOptions(trace_level=tf.RunOptions.HARDWARE_TRACE)
            elif MCd["trace_level"].lower() == "None":
                run_options = None
            else:
                run_options = None
            run_metadata = tf.RunMetadata()
        except KeyError:
            run_options = None
            pass
        logger.debug("trace level set: {}".format(run_options))

        local_step = 0  # This should be an internal tf counter.
        for e in tqdm(range(1, MCd["epochs"] + 1)):
            logger.info("-> START epoch num: {}".format(e))
            t_and_v_reset_ops = [
                val_met_reset_op,
                val_loss_reset_op,
                train_met_reset_op,
                train_loss_reset_op,
            ]
            sess.run(t_and_v_reset_ops)
            logger.debug(
                "reset train and validation metric accumulators: {}".format(
                    t_and_v_reset_ops
                )
            )

            # reinitialize training iterator
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
            sess.run(tr_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})
            next_tr_element = tr_iter.get_next()
            logger.debug("reinitialize training iterator: {}".format(tfr_f_path))

            # main training loop
            logger.debug("-> START iterating training dataset")
            while True:
                try:
                    local_step += 1
                    data, target = sess.run(next_tr_element)
                    # target = np.reshape(target, (target.shape[0], 1))
                    if run_options != None:
                        sess.run(
                            [training_op],
                            feed_dict={X: data, y_raw: target, training: True},
                            options=run_options,
                            run_metadata=run_metadata,
                        )
                        sess.run(
                            [
                                train_auc_update,
                                train_acc_update,
                                train_mean_loss_update,
                            ],
                            feed_dict={X: data, y_raw: target, training: True},
                        )
                    else:
                        sess.run(
                            [
                                training_op,
                                train_auc_update,
                                train_acc_update,
                                train_mean_loss_update,
                            ],
                            feed_dict={X: data, y_raw: target, training: True},
                        )
                    if local_step % 20 == 0:
                        # not sure about this...
                        hist_str = sess.run(hist_write_op)
                        train_writer.add_summary(hist_str, local_step)
                        train_writer.flush()
                except tf.errors.OutOfRangeError:
                    logger.debug("[END] iterating training dataset")
                    break

            # write average for epoch
            summary = sess.run(epoch_train_write_op)
            if run_options != None:
                train_writer.add_run_metadata(run_metadata, "step%d" % e)
            train_writer.add_summary(summary, e)
            train_writer.flush()

            ## validation
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_val"])
            sess.run(val_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})
            next_val_element = val_iter.get_next()
            logger.debug("reinitialize validation iterator: {}".format(tfr_f_path))

            logger.debug("-> START iterating validation dataset")
            while True:
                try:
                    Xb, yb = sess.run(next_val_element)
                    # yb = np.reshape(yb, (yb.shape[0], 1))
                    sess.run(
                        [val_auc_update, val_acc_update, val_mean_loss_update],
                        feed_dict={X: Xb, y_raw: yb},
                    )
                except tf.errors.OutOfRangeError:
                    logger.debug("[END] iterating validation dataset")
                    break

            # check for (and save) best validation params here
            # TODO: there should be a flag here as desired
            cur_loss, cur_acc = sess.run([val_mean_loss, val_acc])
            logger.info("epoch {} validation loss: {}".format(e, cur_loss))
            if cur_loss < best_val_loss:
                last_best_e = e
                best_val_loss = cur_loss
                save_path = saver.save(sess, MCd["saver_save"])
                logger.debug("Model checkpoint saved in path: {}".format(save_path))
                logger.info(
                    "best params saved: val acc: {:.3f}% val loss: {:.4f}".format(
                        cur_acc * 100, cur_loss
                    )
                )

            # Early stopping conditions will start tracking after the WARM_UP_e period
            if EARLY_STOPPING_e > 0:
                if e > WARM_UP_e:
                    if e - last_best_e > EARLY_STOPPING_e:
                        logger.debug(
                            "EARLY STOPPING. val_loss has not improved in {} epochs".format(
                                WARM_UP_e
                            )
                        )
                        break
                else:
                    logger.debug("In warm up period: e {} <= {}".format(e, WARM_UP_e))

            summary = sess.run(epoch_validation_write_op)
            val_writer.add_summary(summary, e)
            val_writer.flush()
            logger.info("[END] epoch num: {}".format(e))

        # TL INIT_DEBUG, post training
        # d1_post = sess.run(dense_1_vars)
        # print(d1_post[AA][BB])

        train_writer.close()
        val_writer.close()
        logger.info("[END] training graph")
    return sess
