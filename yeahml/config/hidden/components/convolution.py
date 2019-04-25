import sys


def configure_conv_layer(opts_raw: dict) -> dict:

    # TODO: get from config?
    CONV_OPTIONS = [
        # "inputs",
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "data_format",
        "dilation_rate",
        # "activation",
        "use_bias",
        "kernel_initializer",
        "bias_initializer",
        "kernel_regularizer",
        "bias_regularizer",
        "activity_regularizer",
        "kernel_constraint",
        "bias_constraint",
        "trainable",
        # "name",
        "reuse",
    ]

    opts_formatted = {}
    for opt in opts_raw:
        if opt not in opts_raw:
            sys.exit()

        # available at https://www.tensorflow.org/api_docs/python/tf/layers/conv2d
        if opt == "filters":
            # TODO: filter logic
            try:
                filters = opts_raw["filters"]
            except KeyError:
                sys.exit("must specify the number of filters: {}".format(opts_raw))
            opts_formatted["filters"] = filters
        elif opt == "kernel_size":
            try:
                kernel_size = opts_raw["kernel_size"]
            except KeyError:
                kernel_size = 3
            opts_formatted["kernel_size"] = filters
        elif opt == "strides":
            try:
                strides = opts_raw["strides"]
            except KeyError:
                strides = 1
            opts_formatted["strides"] = strides
        elif opt == "padding":
            try:
                # TODO: create function for handing bad strings
                padding = opts_raw["padding"]
            except KeyError:
                padding = "SAME"
            opts_formatted["padding"] = padding
        elif opt == "data_format":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                data_format = "channels_last"  # default
            opts_formatted["data_format"] = data_format
        elif opt == "dilation_rate":
            try:
                dilation_rate = opts_raw["dilation_rate"]
            except KeyError:
                dilation_rate = (1, 1)
            opts_formatted["dilation_rate"] = dilation_rate
        elif opt == "use_bias":
            try:
                use_bias = opts_raw["use_bias"]
            except KeyError:
                use_bias = True
            opts_formatted["use_bias"] = use_bias
        elif opt == "kernel_initializer":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "bias_initializer":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "kernel_regularizer":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "bias_regularizer":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "activity_regularizer":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "kernel_constraint":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "bias_constraint":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "trainable":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        elif opt == "reuse":
            try:
                data_format = opts_raw["data_format"]
            except KeyError:
                pass
        else:
            sys.exit("the listed opt: {} does not have an implementation".format(opt))

    try:
        k_init_fn = get_initializer_fn(opts["kernel_initializer"])
    except KeyError:
        k_init_fn = None
    logger.debug("k_init_fn set: {}".format(k_init_fn))

    try:
        k_reg = get_regularizer_fn(opts["kernel_regularizer"])
    except KeyError:
        k_reg = None
    logger.debug("k_reg set: {}".format(k_reg))

    try:
        b_reg = get_regularizer_fn(opts["bias_regularizer"])
    except KeyError:
        b_reg = None
    logger.debug("b_reg set: {}".format(b_reg))

    filters = opts["filters"]
    try:
        kernel_size = opts["kernel_size"]
    except KeyError:
        kernel_size = 3
    logger.debug("kernel_size set: {}".format(kernel_size))

    try:
        # TODO: create func (w/error handling) for this
        padding = opts["padding"]
    except KeyError:
        padding = "SAME"
    logger.debug("padding set: {}".format(padding))

    try:
        strides = opts["strides"]
    except KeyError:
        strides = 1
    logger.debug("strides set: {}".format(strides))

    if not name:
        name = "unnamed_conv_layer"
    logger.debug("name set: {}".format(name))

    try:
        trainable = opts["trainable"]
    except KeyError:
        # trainable by default
        trainable = True
    logger.debug("trainable set: {}".format(trainable))

    return opts_formatted
