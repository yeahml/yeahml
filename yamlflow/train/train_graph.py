import tensorflow as tf
import math
from tqdm import tqdm
import os
import sys

from yamlflow.dataset.handle_data import return_batched_iter  # datasets from tfrecords
from yamlflow.log.yf_logging import config_logger  # custom logging
from yamlflow.build.load_params_onto_layer import init_params_from_file  # load params
from yamlflow.build.get_components import get_run_options
from yamlflow.helper import fmt_metric_summary
from yamlflow.plot.plotting import (
    plot_four_segmentation_array,
    plot_three_segmentation_array,
)


def train_graph(g, MCd: dict, HCd: dict):
    logger = config_logger(MCd, "train")
    logger.info("-> START training graph")

    EARLY_STOPPING_e = MCd["early_stopping_e"]  # default is preset to 0
    WARM_UP_e = MCd["warm_up_epochs"]  # default is 3

    init_global, init_local = g.get_collection("init")
    X, y_raw, training, training_op = g.get_collection("main_ops")
    preds = g.get_collection("preds")

    # performance metrics operations
    train_mets_report, train_mets_update, train_mets_reset_op = g.get_collection(
        "train_metrics"
    )
    val_mets_report, val_mets_update, val_mets_reset = g.get_collection("val_metrics")

    # report loss operations
    train_mean_loss, train_mean_loss_update, train_loss_reset_op = g.get_collection(
        "train_loss"
    )
    val_mean_loss, val_mean_loss_update, val_loss_reset_op = g.get_collection(
        "val_loss"
    )
    t_v_reset_ops = [
        val_mets_reset,
        val_loss_reset_op,
        train_mets_reset_op,
        train_loss_reset_op,
    ]
    epoch_train_write_op, epoch_validation_write_op, hist_write_op = g.get_collection(
        "tensorboard"
    )

    # TODO: TEMP
    if (
        MCd["loss_fn"] == "softmax_binary_segmentation_temp"
        or MCd["loss_fn"] == "softmax_multi_segmentation_temp"
    ):
        y_true_hot = g.get_collection("y_true_hot")
        seg_prob = g.get_collection("seg_prob")

    best_val_loss = math.inf
    last_best_e = 0  # marker for early stopping

    with tf.Session(graph=g) as sess:

        # TODO: temp
        bph = tf.placeholder(dtype=tf.string)
        tfimage = tf.image.decode_png(bph, channels=4)
        # Add the batch dimension
        txi = tf.expand_dims(tfimage, 0)
        # Add image summary
        image_summary_op = tf.summary.image("plot", txi)

        train_writer = tf.summary.FileWriter(
            os.path.join(MCd["log_dir"], "tf_logs", "train"), graph=sess.graph
        )
        val_writer = tf.summary.FileWriter(
            os.path.join(MCd["log_dir"], "tf_logs", "validation")
        )

        sess.run([init_global, init_local])
        saver = tf.train.Saver()  # create after initializing variables

        # initialize variables from file as specified (as used in transfer learning)
        # TODO: I would rather this have a successful return, rather than act as a method
        init_params_from_file(sess, MCd, HCd)

        filenames_ph = tf.placeholder(tf.string, shape=[None])
        tr_iter = return_batched_iter("train", MCd, filenames_ph)
        val_iter = return_batched_iter("val", MCd, filenames_ph)

        # tracing options
        try:
            run_options = get_run_options(MCd["trace_level"])
            run_metadata = tf.RunMetadata()
        except KeyError:
            run_options = None
        logger.debug("trace level set: {}".format(run_options))

        local_step = 0  # This should be an internal tf counter.

        next_tr_element = tr_iter.get_next()
        next_val_element = val_iter.get_next()
        for e in tqdm(range(1, MCd["epochs"] + 1)):
            logger.info("-> START epoch num: {}".format(e))
            sess.run(t_v_reset_ops)
            logger.debug(
                "reset train and validation metric accumulators: {}".format(
                    t_v_reset_ops
                )
            )

            # reinitialize training iterator
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
            sess.run(tr_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})

            logger.debug("reinitialize training iterator: {}".format(tfr_f_path))

            # main training loop
            logger.debug("-> START iterating training dataset")
            while True:
                try:
                    local_step += 1
                    # data, target, _ = sess.run(next_tr_element)
                    data, target = sess.run(next_tr_element)

                    if run_options != None:
                        sess.run(
                            [training_op],
                            feed_dict={X: data, y_raw: target, training: True},
                            options=run_options,
                            run_metadata=run_metadata,
                        )
                        sess.run(
                            [train_mets_update, train_mean_loss_update],
                            feed_dict={X: data, y_raw: target, training: True},
                        )
                    else:
                        sess.run(
                            [training_op, train_mets_update, train_mean_loss_update],
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
            summary_dict = fmt_metric_summary(summary)
            logger.info("epoch {} training metrics: {}".format(e, summary_dict))
            if run_options != None:
                train_writer.add_run_metadata(run_metadata, "step%d" % e)
            train_writer.add_summary(summary, e)
            train_writer.flush()

            ## validation
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_val"])
            sess.run(val_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})
            logger.debug("reinitialize validation iterator: {}".format(tfr_f_path))

            logger.debug("-> START iterating validation dataset")
            while True:
                try:
                    # Xb, yb, _ = sess.run(next_val_element)
                    Xb, yb = sess.run(next_val_element)

                    sess.run(
                        [val_mets_update, val_mean_loss_update],
                        feed_dict={X: Xb, y_raw: yb},
                    )
                except tf.errors.OutOfRangeError:
                    logger.debug("[END] iterating validation dataset")
                    break

            # check for (and save) best validation params here
            # TODO: there should be a flag here as desired
            cur_loss = sess.run(val_mean_loss)
            logger.info("epoch {} validation loss: {}".format(e, cur_loss))
            if cur_loss < best_val_loss:
                last_best_e = e
                best_val_loss = cur_loss
                save_path = saver.save(sess, MCd["saver_save"])
                logger.debug("Model checkpoint saved in path: {}".format(save_path))
                logger.info("best params saved: val loss: {:.4f}".format(cur_loss))

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

            summary_dict = fmt_metric_summary(summary)
            logger.info(
                "[END] epoch num: {} validation metrics: {}".format(e, summary_dict)
            )

            # TODO: TEMP
            if (
                MCd["loss_fn"] == "softmax_binary_segmentation_temp"
                or MCd["loss_fn"] == "softmax_multi_segmentation_temp"
            ):
                ## image
                plot_buf = plot_four_segmentation_array(
                    sess,
                    MCd["output_dim"],
                    X,
                    preds,
                    seg_prob,
                    Xb,
                    yb,
                    idx=0,  # TODO: This is currently hardcoded
                    NUMCLASSES=MCd["num_classes"],
                )

                img_summary = sess.run(
                    image_summary_op, feed_dict={bph: plot_buf.getvalue()}
                )
                val_writer.add_summary(img_summary, e)
                val_writer.flush()

        train_writer.close()
        val_writer.close()
        logger.info("[END] training graph")
    return sess
