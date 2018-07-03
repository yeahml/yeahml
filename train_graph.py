import tensorflow as tf
import math
from tqdm import tqdm
import os

# TODO: make sure global var still works....
from handle_data import return_batched_iter

# from helper import load_obj, save_obj, get_model_params


def train_graph(g, MCd, HCd):

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
    #     next_tr_element, next_val_element, _ = g.get_collection("data_sets")

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

        # ================= transfer learning
        # TODO: if a special path is listed to load vars from for a particular layer,
        # a second init is needed. I don't think this use case will be very common
        # so I will come back and work on this logic later

        # print("TRANSFER INIT : {}".format(MCd["load_params_path"]))
        load_names, layer_tensor_params = [], []
        for i, l_name in enumerate(HCd["layers"]):
            try:
                if HCd["layers"][l_name]["saver"]["load_params"]:
                    # set name to load var from the indicated path, will default to
                    # the current name of the layer
                    try:
                        load_name = HCd["layers"][l_name]["saver"]["load_name"]
                        if load_name == None:
                            load_name = l_name
                    except KeyError:
                        load_name = l_name
                    # set the path from which to load the variables. The default path
                    # is set in the model config but an option is presented to load from other files
                    # try:
                    #    load_path = HCd["layers"][l_name]["saver"]["load_path"]
                    # except KeyError:
                    #    load_name = MCd["load_params_path"]

                    try:
                        name_str = "{}".format(l_name)
                        layer_tensor = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name_str
                        )
                        # print(layer_tensor)
                        # print("len -----", len(layer_tensor))
                        for t_param in layer_tensor:
                            # the split logic is used to remove the (potentially different) name
                            # for example conv_2/kernel:0 will become kernel:0 which will become "kernel", for which we can
                            # append to the name to be used for the layer
                            p_name = t_param.name.split("/")[1].split(":")[0]

                            # build list of vars to load and vars to load onto
                            load_names.append(load_name + "/" + p_name)
                            layer_tensor_params.append(t_param)

                        # layer_tensor = g.get_tensor_by_name("{}:{}".format(l_name))
                    except:
                        sys.exit("unable to get {}".format(l_name))

                    # print("let's do some TL on {}".format(HCd["layers"][l_name]))
            except KeyError:
                # no init from saver
                pass

        assert len(load_names) == len(
            layer_tensor_params
        ), "indicated number of params to load and params found are not equal"
        init_vars = dict(zip(load_names, layer_tensor_params))

        restore_saver = tf.train.Saver(init_vars)
        # print(MCd["load_params_path"])
        restore_saver.restore(sess, MCd["load_params_path"])
        # print(">>>>>>> init_vars restored")
        # ================= [end] transfer learning

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

        local_step = 0  # This should be an internal tf counter.
        for e in tqdm(range(1, MCd["epochs"] + 1)):
            sess.run(
                [
                    val_met_reset_op,
                    val_loss_reset_op,
                    train_met_reset_op,
                    train_loss_reset_op,
                ]
            )

            # reinitialize training iterator
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_train"])
            sess.run(tr_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})
            next_tr_element = tr_iter.get_next()

            # loop entire training set
            # main training loop

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
                    break

            # write average for epoch
            summary = sess.run(epoch_train_write_op)
            if run_options != None:
                train_writer.add_run_metadata(run_metadata, "step%d" % e)
            train_writer.add_summary(summary, e)
            train_writer.flush()

            # run/loop validation
            # reinitialize validation iterator
            tfr_f_path = os.path.join(MCd["TFR_dir"], MCd["TFR_val"])
            sess.run(val_iter.initializer, feed_dict={filenames_ph: [tfr_f_path]})
            next_val_element = val_iter.get_next()
            while True:
                try:
                    Xb, yb = sess.run(next_val_element)
                    # yb = np.reshape(yb, (yb.shape[0], 1))
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
                last_best_e = e
                best_val_loss = cur_loss
                save_path = saver.save(sess, MCd["saver_save"])
                print("Model checkpoint saved in path: %s" % save_path)
                print(
                    "best params saved: val acc: {:.3f}% val loss: {:.4f}".format(
                        cur_acc * 100, cur_loss
                    )
                )
            # Early stopping conditions will start tracking after the WARM_UP_e period
            if EARLY_STOPPING_e > 0:
                if e > WARM_UP_e and e - last_best_e > EARLY_STOPPING_e:
                    # TODO: log early stopping information
                    print("early stopping")
                    break

            summary = sess.run(epoch_validation_write_op)
            val_writer.add_summary(summary, e)
            val_writer.flush()

        train_writer.close()
        val_writer.close()
    return sess
