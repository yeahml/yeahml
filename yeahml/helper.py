import tensorflow as tf


def fmt_tensor_info(t):
    return "| {:15s} | {}".format(t.name.rstrip("0123456789").rstrip(":"), t.shape)


def fmt_metric_summary(summary) -> dict:
    summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary)
    summaries = {}
    for val in summary_proto.value:
        # NOTE: Assuming scalar summaries
        summaries[val.tag] = val.simple_value
    return summaries


def create_layer_names():
    # # TODO: hardcoded way of removing logits from segmentation development
    # if (
    #     MCd["loss_fn"] != "softmax_binary_segmentation_temp"
    #     and MCd["loss_fn"] != "softmax_multi_segmentation_temp"
    # ):
    #     layer_names.append("logits")
    # # exclude all pooling layers
    # # TODO: this logic assumes that the layer name corresponds to the type of layer
    # # > ideally, this list should be built by inspecting the layer 'type', but for beta
    # # > purposes, this works for now.
    # layer_names = [
    #     l
    #     for l in layer_names
    #     if not l.startswith("pool") and not l.startswith("embedding")
    # ]
    # inds_to_add = []
    # for i, l in enumerate(layer_names):
    #     if l.endswith("rnn"):
    #         try:
    #             if HCd["layers"][l]["options"]["condense_out"]:
    #                 inds_to_add.append(i + 1)
    #         except KeyError:
    #             pass

    # # j is needed to ensure the indexes don't become incorrect
    # # after the insertion of each new piece
    # j = 0
    # for i in inds_to_add:
    #     layer_names.insert(i + j, "condense_rnn")
    #     j += 1

    # # multi-layer RNN
    # inds_to_add = []
    # for i, l in enumerate(layer_names):
    #     if l.endswith("rnn"):
    #         try:
    #             num_layers = HCd["layers"][l]["options"]["num_layers"]
    #             if num_layers > 1:
    #                 inds_to_add.append([i + 1, num_layers])
    #         except KeyError:
    #             pass
    # # j is needed to ensure the indexes don't become incorrect
    # # after the insertion of each new piece
    # j = 0
    # for t in inds_to_add:
    #     i = t[0]
    #     for v in range(t[1]):
    #         # add to same index t[1] number of times
    #         if v == 0:
    #             layer_names[i + j - 1] = "layer_{}_rnn".format(v + 1)
    #         else:
    #             layer_names.insert(i + j, "layer_{}_rnn".format(v + 1))
    #             j += 1

    #         # bidirectional RNN
    # inds_to_add = []
    # for i, l in enumerate(layer_names):
    #     if l.endswith("rnn"):
    #         try:
    #             if HCd["layers"][l]["options"]["bidirectional"]:
    #                 inds_to_add.append(i + 1)
    #         except KeyError:
    #             pass
    # # j is needed to ensure the indexes don't become incorrect
    # # after the insertion of each new piece
    # j = 0
    # for i in inds_to_add:
    #     layer_names[i + j] = "bi_rnn_fw"
    #     layer_names.insert(i + j, "bi_rnn_bw")
    #     j += 1

    # # TODO: this is a 'quick and dirty fix' that needs to be thought out....
    # # print([w.name.split("/") for w in weights])
    # weights_adj = []
    # for w in weights:
    #     try:
    #         if w.name.split("/")[-3] == "gru_cell":
    #             if w.name.split("/")[-2] == "candidate":
    #                 pass
    #             else:
    #                 weights_adj.append(w)
    #     except IndexError:
    #         weights_adj.append(w)

    # bias_adj = []
    # for b in bias:
    #     try:
    #         if b.name.split("/")[-3] == "gru_cell":
    #             if b.name.split("/")[-2] == "candidate":
    #                 pass
    #             else:
    #                 bias_adj.append(b)
    #     except IndexError:
    #         bias_adj.append(b)

    # bias = bias_adj
    # weights = weights_adj
    # print(bias)
    # print("---")
    # print(layer_names)
    pass

