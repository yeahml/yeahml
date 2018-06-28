import tensorflow as tf
import numpy as np

from build_hidden import build_hidden_block

# Helper to make the output "consistent"
def reset_graph_deterministic(seed=42):
    # there is no option for deterministic behavior yet...
    # > tf issue https://github.com/tensorflow/tensorflow/issues/18096
    # os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def get_optimizer(MCd: dict):
    opt = MCd["optimizer"].lower()
    optimizer = None
    if opt == "adam":
        optimizer = tf.train.AdamOptimizer(
            learning_rate=MCd["lr"],
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name="Adam",
        )
    elif opt == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=MCd["lr"], name="GradientDescent"
        )
    elif opt == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=MCd["lr"],
            rho=0.95,
            epsilon=1e-08,
            use_locking=False,
            name="Adadelta",
        )
    elif opt == "adagrad":
        optimizer = tf.train.AdagradOptimizer(
            learning_rate=MCd["lr"],
            initial_accumulator_value=0.1,
            use_locking=False,
            name="Adagrad",
        )
    # elif opt == "momentum":
    #     tf.train.MomentumOptimizer(
    #         learning_rate=MCd["lr"],
    #         momentum, # TODO: value
    #         use_locking=False,
    #         name="Momentum",
    #         use_nesterov=False,
    #     )
    elif opt == "ftrl":
        optimizer = tf.train.FtrlOptimizer(
            learning_rate=MCd["lr"],
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0,
            use_locking=False,
            name="Ftrl",
            accum_name=None,
            linear_name=None,
            l2_shrinkage_regularization_strength=0.0,
        )
    elif opt == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=MCd["lr"],
            decay=0.9,
            momentum=0.0,
            epsilon=1e-10,
            use_locking=False,
            centered=False,
            name="RMSProp",
        )
    else:
        # TODO: error handle?
        # realistically this should be caught by the initial check
        pass

    return optimizer


def build_graph(MCd: dict, ACd: dict):

    try:
        reset_graph_deterministic(MCd["seed"])
    except KeyError:
        reset_graph()

    g = tf.Graph()
    with g.as_default():

        #### model architecture
        with tf.name_scope("inputs"):
            # TODO: input dimension logic (currently hardcoded)
            training = tf.placeholder_with_default(False, shape=(), name="training")

            X = tf.placeholder(dtype=tf.float32, shape=(MCd["in_dim"]), name="X_in")
            y_raw = tf.placeholder(
                dtype=tf.int64, shape=(MCd["output_dim"]), name="y_in"
            )
            y = tf.cast(y_raw, tf.float32, name="label")

        hidden = build_hidden_block(X, training, MCd, ACd)

        with tf.name_scope("logits"):
            logits = tf.layers.dense(hidden, MCd["output_dim"][-1], name="logits")
            preds = tf.sigmoid(logits, name="preds")

        #### loss logic
        with tf.name_scope("loss"):
            # Classification, BINARY
            if MCd["final_type"].lower() == "binary":
                xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=y, logits=logits
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
            optimizer = get_optimizer(MCd)
            training_op = optimizer.minimize(batch_loss, name="training_op")

        #### saver
        with tf.name_scope("save_session"):
            init_global = tf.global_variables_initializer()
            init_local = tf.local_variables_initializer()
            saver = tf.train.Saver()

        #### metrics
        with tf.name_scope("metrics"):
            # ================================== performance
            with tf.name_scope("common"):
                # preds = tf.nn.softmax(logits, name="prediction")
                # y_true_cls = tf.argmax(y,1)
                # y_pred_cls = tf.argmax(preds,1)
                y_true_cls = tf.greater_equal(y, 0.5)
                y_pred_cls = tf.greater_equal(preds, 0.5)

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
        for node in (saver, init_global, init_local):
            g.add_to_collection("save_init", node)
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
        weights = [
            v
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if v.name.endswith("kernel:0")
        ]
        bias = [
            v
            for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            if v.name.endswith("bias:0")
        ]
        assert len(weights) == len(bias), "number of weights & bias are not equal"
        layer_names = list(ACd["layers"])
        layer_names.append("logits")
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

    return g
