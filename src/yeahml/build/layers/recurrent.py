import tensorflow as tf

# from yeahml.build.layers.dense import build_dense_layer


# def get_recurrent_cell(opts: dict, logger, g_logger):
#     # TODO: add dtype, add activation
#     try:
#         CELL_TYPE: str = opts["cell_type"].lower()
#     except KeyError:
#         CELL_TYPE = "lstm"
#     logger.debug("cell_type set: {}".format(CELL_TYPE))

#     try:
#         NUM_RNN_NEURONS: int = opts["num_neurons"]
#     except KeyError:
#         NUM_RNN_NEURONS = 3
#     logger.debug("num_neurons set: {}".format(NUM_RNN_NEURONS))

#     RETURN_CELL = None
#     if CELL_TYPE == "lstm":
#         # TODO: There are many othe options that could be implemented
#         # > https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/LSTMCell
#         try:
#             USE_PEEPHOLES: bool = opts["use_peepholes"]
#         except KeyError:
#             USE_PEEPHOLES = False
#         logger.debug("use_peepholes set: {}".format(USE_PEEPHOLES))

#         RETURN_CELL = tf.nn.rnn_cell.LSTMCell(
#             num_units=NUM_RNN_NEURONS, use_peepholes=USE_PEEPHOLES
#         )
#     elif CELL_TYPE == "gru":
#         RETURN_CELL = tf.nn.rnn_cell.GRUCell(num_units=NUM_RNN_NEURONS)
#     elif CELL_TYPE == "basic":
#         RETURN_CELL = tf.nn.rnn_cell.BasicRNNCell(num_units=NUM_RNN_NEURONS)
#     else:
#         sys.exit("cell type ({}) not allowed".format(CELL_TYPE))

#     try:
#         KEEP_PROB: float = opts["keep_prob"]
#     except KeyError:
#         KEEP_PROB = 1.0
#     logger.debug("keep_prob set: {}".format(KEEP_PROB))

#     if KEEP_PROB < 1.0:
#         # TODO: allow for different types of dropout here
#         RETURN_CELL = tf.nn.rnn_cell.DropoutWrapper(
#             RETURN_CELL, input_keep_prob=KEEP_PROB
#         )
#         g_logger.info(
#             ">> dropout: {} (1 - keep_prob({}))".format(1 - KEEP_PROB, KEEP_PROB)
#         )

#     return RETURN_CELL


def build_cell_layer(opts: dict, activation, logger, g_logger):
    # TODO: add dtype, add activation
    try:
        CELL_TYPE: str = opts["cell_type"].lower()
    except KeyError:
        CELL_TYPE = "lstm"
    logger.debug(f"cell_type set: {CELL_TYPE}")

    try:
        NUM_RNN_NEURONS: int = opts["num_neurons"]
    except KeyError:
        NUM_RNN_NEURONS = 3
    logger.debug(f"num_neurons set: {NUM_RNN_NEURONS}")

    # TODO: cell type
    # TODO: params
    # if CELL_TYPE == "lstm":
    out = tf.keras.layers.LSTM(
        NUM_RNN_NEURONS,
        activation=actfn,
        recurrent_activation="sigmoid",
        use_bias=True,
        kernel_initializer="glorot_uniform",
        recurrent_initializer="orthogonal",
        bias_initializer="zeros",
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        implementation=1,
        return_sequences=False,
        return_state=False,
        go_backwards=False,
        stateful=False,
        time_major=False,
        unroll=False,
    )

    return out


def build_recurrent_layer(opts: dict, activation, name: str, logger, g_logger):

    # Check the current input and whether a dim needs to be added
    # e.g. (None, 256) -> (None, 256, 1)
    # NOTE: not sure if get_shape() or .shape is better
    REMOVE_DIM: bool = False
    # cur_input_rank = len(cur_input.get_shape())

    # if cur_input_rank < 3:
    #     # possibly need to add a dim (will need to then remove at the end)
    #     if cur_input.get_shape()[-1] > 1:
    #         logger.debug(
    #             "cur_input shape: ({}) in `build_recurrent_layer`".format(
    #                 cur_input.get_shape()
    #             )
    #         )
    #         cur_input = tf.expand_dims(cur_input, -1)
    #         logger.debug(
    #             "cur_input reshaped: ({}) in `build_recurrent_layer`".format(
    #                 cur_input.get_shape()
    #             )
    #         )
    #         REMOVE_DIM = True
    #     else:
    #         sys.exit(
    #             "The current input is shape: ({}) and is required to be rank > 3".format(
    #                 cur_input.get_shape()
    #             )
    #         )
    #         # this would mean the input is (NONE, 1)... and would need to become
    #         # (None, 1, 1) and I'm not sure this is the case, especially when
    #         # dealing with an RNN

    try:
        NUM_LAYERS: int = opts["num_layers"]
    except KeyError:
        NUM_LAYERS = 1
    logger.debug(f"num_layers set: {NUM_LAYERS}")

    try:
        BIDIRECTIONAL: bool = opts["bidirectional"]
    except KeyError:
        BIDIRECTIONAL = False
    logger.debug(f"bidirectional set: {BIDIRECTIONAL}")

    # try:
    #     DENSE_OPTS_DICT: dict = opts["condense_out"]
    # except KeyError:
    #     DENSE_OPTS_DICT = None
    # logger.debug("condense out: {}".format(DENSE_OPTS_DICT))

    for layer_num in range(1, NUM_LAYERS + 1):
        out = build_cell_layer(opts, actfn, logger, g_logger)

    if BIDIRECTIONAL:
        merge_mode = None  # TODO: include (concat, ave, sum, mul)
        out = tf.keras.layers.Bidirectional(out, merge_mode=merge_mode)

    logger.debug(f"tensor obj: {out}")
    logger.debug(f"[End] building: {name}")

    # if DENSE_OPTS_DICT:
    #     out = build_dense_layer(
    #         out, training, DENSE_OPTS_DICT, None, "rnn_condense", logger, g_logger
    #     )

    # logger.debug(
    #     "out in output shape: ({}) in `build_recurrent_layer`".format(out.get_shape())
    # )

    # if REMOVE_DIM:
    #     out = tf.squeeze(out, -1)
    #     logger.debug(
    #         "out reshaped to final output shape: ({}) in `build_recurrent_layer`".format(
    #             out.get_shape()
    #         )
    #     )
    # g_logger.info("{}".format(fmt_tensor_info(out)))

    return out
