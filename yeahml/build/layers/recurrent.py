import sys
import tensorflow as tf
from yeahml.helper import fmt_tensor_info
from yeahml.build.layers.dense import build_dense_layer


def build_recurrent_layer(
    cur_input, training, opts: dict, actfn, name: str, logger, g_logger
):

    # Check the current input and whether a dim needs to be added
    # e.g. (None, 256) -> (None, 256, 1)
    # NOTE: not sure if get_shape() or .shape is better
    REMOVE_DIM: bool = False
    cur_input_rank = len(cur_input.get_shape())

    if cur_input_rank < 3:
        # possibly need to add a dim (will need to then remove at the end)
        if cur_input.get_shape()[-1] > 1:
            logger.debug(
                "cur_input shape: ({}) in `build_recurrent_layer`".format(
                    cur_input.get_shape()
                )
            )
            cur_input = tf.expand_dims(cur_input, -1)
            logger.debug(
                "cur_input reshaped: ({}) in `build_recurrent_layer`".format(
                    cur_input.get_shape()
                )
            )
            REMOVE_DIM = True
        else:
            sys.exit(
                "The current input is shape: ({}) and is required to be rank > 3".format(
                    cur_input.get_shape()
                )
            )
            # this would mean the input is (NONE, 1)... and would need to become
            # (None, 1, 1) and I'm not sure this is the case, especially when
            # dealing with an RNN

    # TODO: this will need to broken down much further
    # - different cell types
    # - bidirectional
    # - managing outputs vs state

    # forward or bidirectional
    try:
        NUM_LAYERS = opts["num_layers"]
    except KeyError:
        NUM_LAYERS = 1
    logger.debug("num_layers set: {}".format(NUM_LAYERS))

    try:
        BIDIRECTIONAL = opts["bidirectional"]
    except KeyError:
        BIDIRECTIONAL = False
    logger.debug("bidirectional set: {}".format(BIDIRECTIONAL))

    try:
        USE_PEEPHOLES = opts["use_peepholes"]
    except KeyError:
        USE_PEEPHOLES = False
    logger.debug("use_peepholes set: {}".format(USE_PEEPHOLES))

    try:
        NUM_RNN_NEURONS = opts["num_neurons"]
    except KeyError:
        NUM_RNN_NEURONS = 3
    logger.debug("num_neurons set: {}".format(NUM_RNN_NEURONS))

    try:
        dense_ops = opts["condense_out"]
    except KeyError:
        dense_ops = None
    logger.debug("condense out: {}".format(NUM_RNN_NEURONS))

    try:
        KEEP_PROB = opts["keep_prob"]
    except KeyError:
        KEEP_PROB = 0.2
    logger.debug("keep_prob set: {}".format(KEEP_PROB))

    if BIDIRECTIONAL:
        with tf.variable_scope("{}_{}_layer".format(name, NUM_LAYERS)):
            rnn_io = cur_input
            for layer_n in range(NUM_LAYERS):
                with tf.variable_scope(
                    "rnn_layer_{}".format(layer_n), reuse=tf.AUTO_REUSE
                ):
                    ## forward cell
                    # TODO: simple, lstm, gru
                    # TODO: allow for dynamic number of rnn neurons in future?
                    cell_fw = tf.nn.rnn_cell.LSTMCell(
                        num_units=NUM_RNN_NEURONS, use_peepholes=USE_PEEPHOLES
                    )
                    # TODO: allow for different types of dropout here
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                        cell_fw, input_keep_prob=KEEP_PROB
                    )
                    # g_logger.info("fw >> dropout: {}".format(dropout_rate))

                    ## backward cell
                    cell_bw = tf.nn.rnn_cell.LSTMCell(
                        num_units=NUM_RNN_NEURONS, use_peepholes=USE_PEEPHOLES
                    )
                    # TODO: allow for different types of dropout here
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                        cell_bw, input_keep_prob=KEEP_PROB
                    )
                    # g_logger.info("bw >> dropout: {}".format(dropout_rate))

                    outputs, states = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=cell_fw,
                        cell_bw=cell_bw,
                        inputs=rnn_io,
                        dtype=tf.float32,  # TODO: get dtype and set default
                    )

                    # concatenate output (from forward and backward cells)
                    rnn_io = tf.concat(outputs, 2)
    else:
        with tf.variable_scope("{}_{}_layer".format(name, NUM_LAYERS)):
            # rnn_io = cur_input
            layers = []
            for layer_n in range(NUM_LAYERS):
                with tf.variable_scope(
                    "rnn_layer_{}".format(layer_n), reuse=tf.AUTO_REUSE
                ):
                    # TODO: simple, lstm, gru
                    # TODO: allow for dynamic number of rnn neurons in future?
                    cell_fw = tf.nn.rnn_cell.LSTMCell(
                        num_units=NUM_RNN_NEURONS, use_peepholes=USE_PEEPHOLES
                    )
                    # TODO: allow for different types of dropout here
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                        cell_fw, input_keep_prob=KEEP_PROB
                    )
                    layers.append(cell_fw)

            multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(cells=layers)
            outputs, states = tf.nn.dynamic_rnn(
                cell=multi_layer_cell, inputs=cur_input, dtype=tf.float32
            )
            rnn_io = outputs
    out = rnn_io
    logger.debug("tensor obj: {}".format(out))
    logger.debug("[End] building: {}".format(name))

    if dense_ops:
        out = build_dense_layer(
            out, training, dense_ops, None, "rnn_condense", logger, g_logger
        )

    logger.debug(
        "out in output shape: ({}) in `build_recurrent_layer`".format(out.get_shape())
    )

    if REMOVE_DIM:
        out = tf.squeeze(out, -1)
        logger.debug(
            "out reshaped to final output shape: ({}) in `build_recurrent_layer`".format(
                out.get_shape()
            )
        )
    g_logger.info("{}".format(fmt_tensor_info(out)))

    return out
