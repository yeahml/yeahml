import tensorflow as tf
import sys

# import numpy as np # flatten op

from yamlflow.log.yf_logging import config_logger
from yamlflow.build.build_hidden import build_hidden_block
from yamlflow.build.get_components import get_tf_dtype
from yamlflow.build.get_components import get_optimizer
from yamlflow.build.helper import build_mets_write_op
from yamlflow.helper import fmt_tensor_info


# Helper to make the output "consistent"
def reset_graph_deterministic(seed=42):
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph_deterministic")
    # there is no option for deterministic behavior yet...
    # > tf issue https://github.com/tensorflow/tensorflow/issues/18096
    # os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    # np.random.seed(seed)


def reset_graph(seed=42):
    # logger = logging.getLogger("build_logger")
    # logger.info("reset_graph")
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    # np.random.seed(seed)


def build_graph(MCd: dict, HCd: dict):
    # logger = logging.getLogger("build_logger")
    logger = config_logger(MCd, "build")
    logger.info("-> START building graph")

    try:
        reset_graph_deterministic(MCd["seed"])
    except KeyError:
        reset_graph()

    g_logger = config_logger(MCd, "graph")

    g = tf.Graph()
    with g.as_default():

        g_logger.info("============={}=============".format("GRAPH"))
        #### model architecture
        with tf.name_scope("inputs"):
            logger.info("create inputs")
            # TODO: input dimension logic (currently hardcoded)
            training = tf.placeholder_with_default(False, shape=(), name="training")

            # raw data input dim/type
            # TODO: consider x_raw, similar to y_raw. will need to update collection
            x_dtype = get_tf_dtype(MCd["in_dtype"])
            X = tf.placeholder(dtype=x_dtype, shape=(MCd["in_dim"]), name="X_in")
            if MCd["reshape_in_to"]:
                X = tf.reshape(X, shape=(MCd["reshape_in_to"]), name="data")
            g_logger.info("{}".format(fmt_tensor_info(X)))

            # label input dim/type
            label_dtype = get_tf_dtype(MCd["label_dtype"])
            y_raw = tf.placeholder(
                dtype=label_dtype, shape=(MCd["output_dim"]), name="y_in"
            )

            # TODO: revamp this conversion. I think for metrics they will need to be
            # consistent, but in general, this should be handled more carefully as it
            # will break in certain situations.
            # > int64 probably shouldn't be mapped to float32
            if MCd["label_dtype"].startswith("int"):
                y = tf.cast(y_raw, tf.float32, name="label")
            else:
                # this is currently needed so that the raw values can be added to a collection
                # and retrieved later for both converted and not converted values
                y = y_raw

        hidden = build_hidden_block(X, training, MCd, HCd, logger, g_logger)

        # this will act as the output layer for regression
        logits = tf.layers.dense(hidden, MCd["output_dim"][-1], name="logits")

        # TODO: there are now three locations where the type of problem alters code choices
        if MCd["loss_fn"] == "sigmoid":
            preds = tf.sigmoid(logits, name="y_proba")
        elif MCd["loss_fn"] == "softmax":
            preds = tf.nn.softmax(logits, name="y_proba")
        elif MCd["loss_fn"] == "mse" or MCd["loss_fn"] == "rmse":
            preds = logits
        else:
            logger.fatal("preds cannot be created as: {}".format(MCd["loss_fn"]))
            sys.exit(
                "final_type: {} -- is not supported or defined.".format(MCd["loss_fn"])
            )
        logger.debug("pred created as {}: {}".format(MCd["loss_fn"], preds))

        g_logger.info("{}".format(fmt_tensor_info(preds)))

        #### loss logic
        with tf.name_scope("loss"):
            logger.info("create /loss")
            # TODO: the type of xentropy should be defined in the config
            # > there should also be a check for the type that should be used.
            if MCd["loss_fn"] == "sigmoid":
                xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=y
                )
            elif MCd["loss_fn"] == "softmax":
                # why v2? see here: https://bit.ly/2z3NJ8n
                # xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                #    logits=logits, labels=y
                # )
                xentropy = tf.losses.softmax_cross_entropy(
                    onehot_labels=y, logits=logits
                )
            elif MCd["loss_fn"] == "mse" or MCd["loss_fn"] == "rmse":
                xentropy = tf.losses.mean_squared_error(labels=y, predictions=preds)
            else:
                logger.fatal("xentropy cannot be created as: {}".format(MCd["loss_fn"]))
                sys.exit(
                    "final_type: {} -- is not supported or defined.".format(
                        MCd["loss_fn"]
                    )
                )
            logger.debug("xentropy created as {}: {}".format(MCd["loss_fn"], xentropy))

            base_loss = tf.reduce_mean(xentropy, name="base_loss")
            # handle regularization losses
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # TODO: ensure this returns 0/None when regularization is not used
            # reg_losses = tf.losses.get_regularization_loss(
            # scope=None, name="total_regularization_loss"
            # )
            batch_loss = tf.add_n([base_loss] + reg_losses, name="loss")

        #### optimizer
        with tf.name_scope("train"):
            logger.info("create /train")
            optimizer = get_optimizer(MCd)
            g_logger.info("{}".format(optimizer._name))

            training_op = optimizer.minimize(batch_loss, name="training_op")

        #### init
        with tf.name_scope("init"):
            logger.info("create /init")
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()

        #### metrics
        with tf.name_scope("metrics"):
            logger.info("create /metrics")
            if MCd["metrics_type"] == "classification":
                # classification performance
                with tf.name_scope("common"):
                    logger.debug("create /metrics/common")
                    if MCd["loss_fn"] == "sigmoid":
                        y_true = tf.greater_equal(y, 0.5)
                        y_pred = tf.greater_equal(preds, 0.5)
                    elif MCd["loss_fn"] == "softmax":
                        y_true = tf.argmax(y, 1)
                        y_pred = tf.argmax(preds, 1)
            else:
                y_true = y
                y_pred = preds

            # TODO: these three metric creations (tr/val/test) could likely be made
            # in a single function with a param for "type". This would prevent any accidental
            # inconsistencies when creating metrics for each type
            # TODO: groups may be able to be made dynamically by appending() all the report and
            # and update operations to a lists which are then passed to the group

            #### Train
            train_mets_report_ops = []
            train_mets_update_ops = []
            with tf.name_scope("train_metrics") as scope:
                if MCd["metrics_type"] == "classification":
                    logger.debug("create /metrics/train_metrics [classification]")
                    # AUC
                    train_auc, train_auc_update = tf.metrics.auc(
                        labels=y, predictions=preds
                    )
                    train_mets_report_ops.append(train_auc)
                    train_mets_update_ops.append(train_auc_update)

                    # Accuracy
                    train_acc, train_acc_update = tf.metrics.accuracy(
                        labels=y_true, predictions=y_pred
                    )
                    train_mets_report_ops.append(train_acc)
                    train_mets_update_ops.append(train_acc_update)

                elif MCd["metrics_type"] == "regression":
                    logger.debug("create /metrics/train_metrics [regression]")

                    # RMSE
                    train_rmse, train_rmse_update = tf.metrics.root_mean_squared_error(
                        labels=y, predictions=preds
                    )
                    train_mets_report_ops.append(train_rmse)
                    train_mets_update_ops.append(train_rmse_update)

                else:
                    # although the error should be caught in the config. the exit error
                    # is kept until the supported types are pulled from in a config file
                    # rather than being hardcoded as a list in config.py
                    sys.exit(
                        "train metrics type {} is unsupported".format(
                            MCd["metrics_type"]
                        )
                    )
                # Group metrics
                train_mets_report = tf.group(train_mets_report_ops)
                train_mets_update = tf.group(train_mets_update_ops)
                train_met_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                train_mets_reset = tf.variables_initializer(
                    train_met_vars, name="train_mets_reset"
                )

            #### Validation
            val_mets_report_ops = []
            val_mets_update_ops = []
            with tf.name_scope("val_metrics") as scope:
                if MCd["metrics_type"] == "classification":
                    logger.debug("create /metrics/val_metrics [classification]")

                    # AUC
                    val_auc, val_auc_update = tf.metrics.auc(
                        labels=y, predictions=preds
                    )
                    val_mets_report_ops.append(val_auc)
                    val_mets_update_ops.append(val_auc_update)

                    # Accuracy
                    val_acc, val_acc_update = tf.metrics.accuracy(
                        labels=y_true, predictions=y_pred
                    )
                    val_mets_report_ops.append(val_acc)
                    val_mets_update_ops.append(val_acc_update)

                elif MCd["metrics_type"] == "regression":
                    logger.debug("create /metrics/val_metrics [regression]")

                    # RMSE
                    val_rmse, val_rmse_update = tf.metrics.root_mean_squared_error(
                        labels=y, predictions=preds
                    )
                    val_mets_report_ops.append(val_rmse)
                    val_mets_update_ops.append(val_rmse_update)
                else:
                    # although the error should be caught in the config. the exit error
                    # is kept until the supported types are pulled from in a config file
                    # rather than being hardcoded as a list in config.py
                    sys.exit(
                        "val metrics type {} is unsupported".format(MCd["metrics_type"])
                    )

                # Group metrics
                val_mets_report = tf.group(val_mets_report_ops)
                val_mets_update = tf.group(val_mets_update_ops)
                val_acc_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                val_mets_reset = tf.variables_initializer(
                    val_acc_vars, name="val_mets_reset"
                )

            #### Test
            test_mets_report_ops = []
            test_mets_update_ops = []
            with tf.name_scope("test_metrics") as scope:
                if MCd["metrics_type"] == "classification":
                    logger.debug("create /metrics/test_metrics [classification]")

                    # AUC
                    test_auc, test_auc_update = tf.metrics.auc(
                        labels=y, predictions=preds
                    )
                    test_mets_report_ops.append(test_auc)
                    test_mets_update_ops.append(test_auc_update)

                    # ACC
                    test_acc, test_acc_update = tf.metrics.accuracy(
                        labels=y_true, predictions=y_pred
                    )
                    test_mets_report_ops.append(test_acc)
                    test_mets_update_ops.append(test_acc_update)

                elif MCd["metrics_type"] == "regression":
                    logger.debug("create /metrics/test_metrics [regression]")

                    # RMSE
                    test_rmse, test_rmse_update = tf.metrics.root_mean_squared_error(
                        labels=y, predictions=preds
                    )
                    test_mets_report_ops.append(test_rmse)
                    test_mets_update_ops.append(test_rmse_update)

                else:
                    # although the error should be caught in the config. the exit error
                    # is kept until the supported types are pulled from in a config file
                    # rather than being hardcoded as a list in config.py
                    sys.exit(
                        "test metrics type {} is unsupported".format(
                            MCd["metrics_type"]
                        )
                    )
                test_mets_report = tf.group(test_mets_report_ops)
                test_mets_update = tf.group(test_mets_update_ops)
                test_acc_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                test_mets_reset = tf.variables_initializer(
                    test_acc_vars, name="test_mets_reset"
                )

            # =============================================== loss
            with tf.name_scope("train_loss_eval") as scope:
                logger.debug("create /metrics/train_loss_eval")
                train_mean_loss, train_mean_loss_update = tf.metrics.mean(batch_loss)
                train_loss_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                train_loss_reset_op = tf.variables_initializer(
                    train_loss_vars, name="train_loss_reset_op"
                )
            with tf.name_scope("val_loss_eval") as scope:
                logger.debug("create /metrics/val_loss_eval")
                val_mean_loss, val_mean_loss_update = tf.metrics.mean(batch_loss)
                val_loss_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                val_loss_reset_op = tf.variables_initializer(
                    val_loss_vars, name="val_loss_reset_op"
                )
            with tf.name_scope("test_loss_eval") as scope:
                logger.debug("create /metrics/test_loss_eval")
                test_mean_loss, test_mean_loss_update = tf.metrics.mean(batch_loss)
                test_loss_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                test_loss_reset_op = tf.variables_initializer(
                    test_loss_vars, name="test_loss_rest_op"
                )

        # --- create collections
        for node in (init_global, init_local):
            g.add_to_collection("init", node)
        for node in (X, y_raw, training, training_op):
            g.add_to_collection("main_ops", node)

        g.add_to_collection("preds", preds)

        for node in (y_true, y_pred):
            g.add_to_collection("gt_and_pred", node)

        # performance metrics operations
        for node in (train_mets_report, train_mets_update, train_mets_reset):
            g.add_to_collection("train_metrics", node)
        for node in (val_mets_report, val_mets_update, val_mets_reset):
            g.add_to_collection("val_metrics", node)
        for node in (test_mets_report, test_mets_update, test_mets_reset):
            g.add_to_collection("test_metrics", node)

        # loss metrics operations
        for node in (train_mean_loss, train_mean_loss_update, train_loss_reset_op):
            g.add_to_collection("train_loss", node)
        for node in (val_mean_loss, val_mean_loss_update, val_loss_reset_op):
            g.add_to_collection("val_loss", node)
        for node in (test_mean_loss, test_mean_loss_update, test_loss_reset_op):
            g.add_to_collection("test_loss", node)

        # logits
        g.add_to_collection("logits", logits)

        # ===================================== tensorboard
        #### scalar
        # TODO: this should really be TRAINABLE_VARIABLES...
        weights = [
            v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v.name.rstrip("0123456789").endswith("kernel:")
        ]
        logger.debug("create scalar weights: {}".format(weights))
        bias = [
            v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v.name.rstrip("0123456789").endswith("bias:")
        ]
        logger.debug("create scalar bias: {}".format(weights))
        assert len(weights) == len(bias), "number of weights & bias are not equal"
        logger.debug("len(weights) == len(bias) = {}".format(len(weights)))
        layer_names = list(HCd["layers"])
        layer_names.append("logits")
        # exclude all pooling layers
        # TODO: this logic assumes that the layer name corresponds to the type of layer
        # > ideally, this list should be built by inspecting the layer 'type', but for beta
        # > purposes, this works for now.
        layer_names = [l for l in layer_names if not l.startswith("pool")]
        assert len(weights) == len(layer_names), "num of w&b not equal to num layers"

        hist_list = []
        for i, l_weight in enumerate(weights):
            l_name = layer_names[i] + "_params"
            w_name, b_name = "weights", "bias"
            with tf.variable_scope(l_name):
                w_hist = tf.summary.histogram(w_name, l_weight)
                b_hist = tf.summary.histogram(b_name, bias[i])
                hist_list.append(w_hist)
                hist_list.append(b_hist)

        hist_write_op = tf.summary.merge(hist_list, name="histogram_write_op")
        logger.debug("{} hist opts written".format(hist_list))

        # TODO: would like to combine val+train on the same graph
        # build metrics write op for tensorboard + reporting
        # test is only used for .evaluation, not currently added to TensorBoard
        train_write_op = build_mets_write_op(
            train_mets_report_ops, train_mean_loss, "train"
        )
        val_write_op = build_mets_write_op(val_mets_report_ops, val_mean_loss, "val")
        test_write_op = build_mets_write_op(
            test_mets_report_ops, test_mean_loss, "test"
        )

        # tensorboard collections
        for node in (train_write_op, val_write_op, hist_write_op):
            g.add_to_collection("tensorboard", node)
        g.add_to_collection("tensorboard_test", test_write_op)

    g_logger.info("=============={}==============".format("END"))
    logger.info("[END] building graph")

    return g
