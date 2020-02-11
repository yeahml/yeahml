from yeahml.config.default.config_types import optional_config, parameter_config


def _get_unspecified_default(DEFAULT):
    formatted_dict = {}
    for k, spec in DEFAULT.items():
        # get default values
        o = spec()
        formatted_dict[k] = o
    return formatted_dict


def _get_unspecified_default_spec(DEFAULT):
    formatted_dict = {}
    for k, spec in DEFAULT.items():
        # get default spec
        formatted_dict[k] = spec
    return formatted_dict


def _unpack_params(raw_config, k, spec):
    ret_dict = {}
    known_params = spec.known_dict
    if known_params:
        for kparam_k, kparam_spec in known_params.items():
            out_val = parse_default(raw_config, known_params)
            ret_dict = {**ret_dict, **out_val}

    # obtain all user defined params
    # unsure if this could be checked against the object it is specifying
    # parameters for.
    for k, v in raw_config.items():
        if k in ret_dict:
            pass
        else:
            ret_dict[k] = v
    return ret_dict


def parse_default(raw_config, DEFAULT):
    formatted_dict = {}
    for k, spec in DEFAULT.items():
        if isinstance(spec, optional_config):
            # call optional_config to return dict
            try:
                cur = raw_config[k]
            except KeyError:
                cur = None
            if cur:
                # get the spec and check the params
                default_spec = _get_unspecified_default_spec(spec())
                out_val = parse_default(raw_config[k], default_spec)
            else:
                # no user values are specified, only get the default values
                out_val = _get_unspecified_default(spec())
        elif isinstance(spec, parameter_config):
            out_val = _unpack_params(raw_config[k], k, spec)
        elif callable(spec):
            try:
                raw_val = raw_config[k]
            except KeyError:
                raw_val = None
            # raw_val defaults to None, required will be checked in spec
            out_val = spec(raw_val)
        else:
            out_val = parse_default(raw_config[k], DEFAULT[k])
        formatted_dict[k] = out_val
    return formatted_dict
