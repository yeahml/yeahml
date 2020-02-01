def parse_default(raw_config, DEFAULT):
    formatted_dict = {}
    for k, spec in DEFAULT.items():
        if callable(spec):
            try:
                raw_val = raw_config[k]
            except KeyError:
                raw_val = None
            # raw_val defaults to None, required will be checked in spec
            out_val = spec(raw_val)
        else:
            out_val = parse_default(raw_config, DEFAULT[k])
        formatted_dict[k] = out_val
    return formatted_dict
