import tensorflow as tf

# import numpy as np
import sys

# import custom logging
from yf_logging import config_logger

from build_hidden import build_hidden_block
from get_components import get_tf_dtype

# print information about the graph
from helper import print_tensor_info

# get tf optimizer
from get_components import get_optimizer


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
    logger.info("build_graph")

    try:
        reset_graph_deterministic(MCd["seed"])
    except KeyError:
        reset_graph()

    # G_PRINT is used as bool to determine whether information
    # about the graph should be printed
    try:
        G_PRINT = MCd["print_g_spec"]
    except KeyError:
        G_PRINT = False
    try:
        G_NAME = MCd["name"]
    except KeyError:
        G_NAME = "custom architecture"
    if G_PRINT:
        print("========================{}========================".format(G_NAME))

    g = tf.Graph()
    with g.as_default():

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
            if G_PRINT:
                print_tensor_info(X)

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
        logger.info("create hidden block")
        hidden = build_hidden_block(X, training, MCd, HCd)

        logits = tf.layers.dense(hidden, MCd["output_dim"][-1], name="logits")

        # TODO: there are now three locations where the type of problem alters code choices
        if MCd["final_type"] == "sigmoid":
            preds = tf.sigmoid(logits, name="y_proba")
        elif MCd["final_type"] == "softmax":
            preds = tf.nn.softmax(logits, name="y_proba")
        else:
            sys.exit(
                "final_type: {} -- is not supported or defined.".format(
                    MCd["final_type"]
                )
            )

        if G_PRINT:
            print_tensor_info(preds)

        #### loss logic
        with tf.name_scope("loss"):
            logger.info("create loss")
            # TODO: the type of xentropy should be defined in the config
            # > there should also be a check for the type that should be used.
            if MCd["final_type"] == "sigmoid":
                xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=y
                )
            elif MCd["final_type"] == "softmax":
                # why v2? see here: https://bit.ly/2z3NJ8n
                xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=y
                )
            else:
                sys.exit(
                    "final_type: {} -- is not supported or defined.".format(
                        MCd["final_type"]
                    )
                )

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
            logger.info("training params")
            optimizer = get_optimizer(MCd)
            if G_PRINT:
                print("opt: {}".format(optimizer._name))
            training_op = optimizer.minimize(batch_loss, name="training_op")

        #### init
        with tf.name_scope("init"):
            logger.info("create init")
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()

        #### metrics
        with tf.name_scope("metrics"):
            # ================================== performance
            with tf.name_scope("common"):
                if MCd["final_type"] == "sigmoid":
                    y_true_cls = tf.greater_equal(y, 0.5)
                    y_pred_cls = tf.greater_equal(preds, 0.5)
                elif MCd["final_type"] == "softmax":
                    y_true_cls = tf.argmax(y, 1)
                    y_pred_cls = tf.argmax(preds, 1)

                correct_prediction = tf.equal(
                    y_pred_cls, y_true_cls, name="correct_predictions"
                )
                batch_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            with tf.name_scope("train_metrics") as scope:
                train_auc, train_auc_update = tf.metrics.auc(
                    labels=y, predictions=preds
                )
                train_acc, train_acc_update = tf.metrics.accuracy(
                    labels=y_true_cls, predictions=y_pred_cls
                )
                train_acc_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                train_met_reset_op = tf.variables_initializer(
                    train_acc_vars, name="train_met_reset_op"
                )
            with tf.name_scope("val_metrics") as scope:
                val_auc, val_auc_update = tf.metrics.auc(labels=y, predictions=preds)
                val_acc, val_acc_update = tf.metrics.accuracy(
                    labels=y_true_cls, predictions=y_pred_cls
                )
                val_acc_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                val_met_reset_op = tf.variables_initializer(
                    val_acc_vars, name="val_met_reset_op"
                )
            with tf.name_scope("test_metrics") as scope:
                test_auc, test_auc_update = tf.metrics.auc(labels=y, predictions=preds)
                test_acc, test_acc_update = tf.metrics.accuracy(
                    labels=y_true_cls, predictions=y_pred_cls
                )
                test_acc_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                test_acc_reset_op = tf.variables_initializer(
                    test_acc_vars, name="test_met_reset_op"
                )

            # =============================================== loss
            with tf.name_scope("train_loss_eval") as scope:
                train_mean_loss, train_mean_loss_update = tf.metrics.mean(batch_loss)
                train_loss_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                train_loss_reset_op = tf.variables_initializer(
                    train_loss_vars, name="train_loss_reset_op"
                )
            with tf.name_scope("val_loss_eval") as scope:
                val_mean_loss, val_mean_loss_update = tf.metrics.mean(batch_loss)
                val_loss_vars = tf.contrib.framework.get_variables(
                    scope, collection=tf.GraphKeys.LOCAL_VARIABLES
                )
                val_loss_reset_op = tf.variables_initializer(
                    val_loss_vars, name="val_loss_reset_op"
                )
            with tf.name_scope("test_loss_eval") as scope:
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
        for node in (preds, y_true_cls, y_pred_cls, correct_prediction):
            g.add_to_collection("preds", node)
        for node in (
            train_auc,
            train_auc_update,
            train_acc,
            train_acc_update,
            train_met_reset_op,
        ):
            g.add_to_collection("train_metrics", node)
        for node in (
            val_auc,
            val_auc_update,
            val_acc,
            val_acc_update,
            val_met_reset_op,
        ):
            g.add_to_collection("val_metrics", node)
        for node in (
            test_auc,
            test_auc_update,
            test_acc,
            test_acc_update,
            test_acc_reset_op,
        ):
            g.add_to_collection("test_metrics", node)
        for node in (train_mean_loss, train_mean_loss_update, train_loss_reset_op):
            g.add_to_collection("train_loss", node)
        for node in (val_mean_loss, val_mean_loss_update, val_loss_reset_op):
            g.add_to_collection("val_loss", node)
        for node in (test_mean_loss, test_mean_loss_update, test_loss_reset_op):
            g.add_to_collection("test_loss", node)
        g.add_to_collection("logits", logits)

        # ===================================== tensorboard
        #### scalar
        # TODO: this should really be TRAINABLE_VARIABLES...
        weights = [
            v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v.name.rstrip("0123456789").endswith("kernel:")
        ]
        bias = [
            v
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            if v.name.rstrip("0123456789").endswith("bias:")
        ]
        assert len(weights) == len(bias), "number of weights & bias are not equal"
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
        epoch_train_loss_scalar = tf.summary.scalar("loss/train", train_mean_loss)
        epoch_train_acc_scalar = tf.summary.scalar("acc/train", train_acc)
        epoch_train_auc_scalar = tf.summary.scalar("auc/train", train_auc)
        epoch_train_write_op = tf.summary.merge(
            [epoch_train_loss_scalar, epoch_train_acc_scalar, epoch_train_auc_scalar],
            name="epoch_train_write_op",
        )

        # ===== epoch, validation
        epoch_validation_loss_scalar = tf.summary.scalar("loss/val", val_mean_loss)
        epoch_validation_acc_scalar = tf.summary.scalar("acc/val", val_acc)
        epoch_validation_auc_scalar = tf.summary.scalar("auc/val", val_auc)
        epoch_validation_write_op = tf.summary.merge(
            [
                epoch_validation_loss_scalar,
                epoch_validation_acc_scalar,
                epoch_validation_auc_scalar,
            ],
            name="epoch_validation_write_op",
        )

        for node in (epoch_train_write_op, epoch_validation_write_op, hist_write_op):
            g.add_to_collection("tensorboard", node)

    if G_PRINT:
        print(
            "======================={}=========================".format(
                str(len(G_NAME) * "=")
            )
        )

    return g
