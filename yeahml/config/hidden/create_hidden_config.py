import sys

from yeahml.config.helper import parse_yaml_from_path
from yeahml.config.hidden.components.pooling import configure_pooling_layer
from yeahml.config.hidden.components.convolution import configure_conv_layer


def _get_hidden_layers(h_raw_config: dict):
    try:
        hidden_layers = h_raw_config["layers"]
    except KeyError:
        sys.exit("No Layers Found")

    return hidden_layers


def parse_layer_type_information(hl: dict, default_activation: str) -> dict:
    # parse layer specific information and ensure the options are
    # "reasonable" for each layer
    # set "type", "options", and "activation"
    HLD = {}

    try:
        hl_type = hl["type"].lower()
        HLD["type"] = hl_type
    except KeyError:
        sys.exit("layer does not have a 'type': {}".format(hl))

    ## option logic for each layer type
    # see if options exist
    try:
        opts_raw = hl["options"]
    except:
        # will default to default options
        opts_raw = None

    # opts_formatted = {}
    # if hl_type == "conv2d":
    #     opts_formatted = configure_conv_layer(opts_raw)
    # elif hl_type == "deconv2d":
    #     opts_formatted = configure_conv_layer(opts_raw)
    # elif hl_type == "deconv2d":
    #     pass
    # elif hl_type == "dense":
    #     pass
    # elif hl_type == "deconv2d":
    #     pass
    # elif hl_type == "pooling2d":
    #     opts_formatted = configure_pooling_layer(opts_raw)
    # elif hl_type == "pooling1d":
    #     opts_formatted = configure_pooling_layer(opts_raw)
    # elif hl_type == "global_pooling":
    #     pass
    # elif hl_type == "embedding":
    #     pass
    # elif hl_type == "batch_normalization":
    #     pass
    # elif hl_type == "recurrent":
    #     pass
    # else:
    #     sys.exit("layer type {} not currently supported".format(hl_type))
    HLD["options"] = opts_raw  # opts_formatted

    # TODO: this needs to be pushed down a level since some layers doesn't require act
    try:
        actfn_str = hl["activation"]
    except KeyError:
        try:
            # fall back to the default activation function specified
            actfn_str = default_activation
        except KeyError:
            # relu: the reasoning here is that the relu is subjectively the most
            # common/default activation function in DNNs, but I don't LOVE this
            actfn_str = "relu"
    # TODO: logger.debug("activation set: {}".format(actfn_str))
    HLD["activation"] = actfn_str

    return HLD


def create_layer_config(hl: dict, default_activation: str) -> dict:
    HLD = {}

    ## TODO: parse layer type information
    HLD = parse_layer_type_information(hl, default_activation)

    return HLD


def extract_hidden_dict_and_set_defaults(
    h_raw_config: dict, default_activation: str
) -> dict:
    parsed_h_config = {}
    # create architecture config

    hidden_layers = _get_hidden_layers(h_raw_config)
    if not hidden_layers:
        sys.exit("a hidden layer field was found, but did not contain any layers")

    approved_layers_config = {}
    for hl in hidden_layers:
        approved_layers_config[hl] = create_layer_config(
            hidden_layers[hl], default_activation
        )

    parsed_h_config["layers"] = approved_layers_config

    return parsed_h_config
