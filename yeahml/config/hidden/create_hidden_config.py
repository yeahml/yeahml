from yeahml.config.helper import parse_yaml_from_path
import sys


def _get_hidden_layers(h_raw_config: dict):
    try:
        hidden_layers = h_raw_config["layers"]
    except KeyError:
        sys.exit("No Layers Found")

    return hidden_layers


def parse_layer_type_information(hl: dict, default_activation: str) -> dict:
    # TODO: placeholder for parsing layer options
    # this function should parse layer specific information
    # and ensure the options are "reasonable" for each layer

    HLD = hl

    # see if options exist
    try:
        opts = hl["options"]
    except:
        # TODO: implement (specific to layer type)
        # TODO: these defaults (if specified as None are currently implemented in "build_layers")
        opts = None
        # sys.exit("No options specified for {}".format(hl))
    HLD["options"] = opts

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

    # Read from config file?
    allowed_layer_types = set(
        [
            "conv2d",
            "deconv2d",
            "dense",
            "pooling2d",
            "pooling1d",
            "global_pooling",
            "embedding",
            "batch_normalization",
            "recurrent",
        ]
    )

    # make sure a type is specified
    try:
        hl_type = hl["type"].lower()
    except KeyError:
        sys.exit("layer does not have a 'type': {}".format(hl))

    if hl_type not in allowed_layer_types:
        sys.exit("layer type {} not currently supported".format(hl["type"]))

    # TODO: parse layer type information
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
