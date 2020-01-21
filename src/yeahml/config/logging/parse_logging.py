def format_logging_config(raw_config):
    ####### Logging
    ERR_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    formatted_dict = {}
    ## console
    try:
        temp_c_lvl = raw_config["console"]["level"]
        if not temp_c_lvl:
            # handle null case
            formatted_dict["log_c_lvl"] = "CRITICAL"
        else:
            temp_c_lvl = temp_c_lvl.upper()
            if temp_c_lvl not in ERR_LEVELS:
                raise ValueError(
                    f"console level {temp_c_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                formatted_dict["log_c_lvl"] = temp_c_lvl
    except KeyError:
        formatted_dict["log_c_lvl"] = "CRITICAL"

    try:
        formatted_dict["log_c_str"] = raw_config["console"]["format_str"]
        if not formatted_dict["log_c_str"]:
            # handle null case
            formatted_dict["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        formatted_dict["log_c_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## file
    try:
        temp_f_lvl = raw_config["file"]["level"]
        if not temp_f_lvl:
            # handle null case
            formatted_dict["log_f_lvl"] = "CRITICAL"
        else:
            temp_f_lvl = temp_f_lvl.upper()
            if temp_f_lvl not in ERR_LEVELS:
                raise ValueError(
                    "console level {temp_f_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                formatted_dict["log_f_lvl"] = temp_f_lvl.upper()
    except KeyError:
        formatted_dict["log_f_lvl"] = "CRITICAL"

    try:
        formatted_dict["log_f_str"] = raw_config["file"]["format_str"]
        if not formatted_dict["log_f_str"]:
            # handle null case
            formatted_dict[
                "log_f_str"
            ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"
        else:
            # TODO: error checking
            pass
    except KeyError:
        formatted_dict[
            "log_f_str"
        ] = "[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s()][%(levelname)-8s]: %(message)s"

    ## graph level
    try:
        temp_g_lvl = raw_config["graph_spec"]
        if temp_g_lvl == True:
            # support for a simple bool config
            formatted_dict["log_g_lvl"] = "DEBUG"
        elif not temp_g_lvl:
            # handle null case
            formatted_dict["log_g_lvl"] = "DEBUG"
        else:
            temp_g_lvl = temp_g_lvl.upper()
            if temp_g_lvl not in ERR_LEVELS:
                raise ValueError(
                    "log level {temp_g_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                formatted_dict["log_g_lvl"] = temp_g_lvl
    except KeyError:
        formatted_dict["log_g_lvl"] = "DEBUG"
    # hard set the graph info
    formatted_dict["log_g_str"] = "%(name)-12s: %(levelname)-8s %(message)s"

    ## preds level
    try:
        temp_p_lvl = raw_config["graph_spec"]
        if temp_p_lvl == True:
            # support for a simple bool config
            formatted_dict["log_p_lvl"] = "DEBUG"
        elif not temp_p_lvl:
            # handle null case
            formatted_dict["log_p_lvl"] = "DEBUG"
        else:
            temp_p_lvl = temp_p_lvl.upper()
            if temp_p_lvl not in ERR_LEVELS:
                raise ValueError(
                    "log level {temp_p_lvl} not allowed. please select one of {ERR_LEVELS}"
                )
            else:
                formatted_dict["log_p_lvl"] = temp_p_lvl
    except KeyError:
        formatted_dict["log_p_lvl"] = "DEBUG"
    # hard set the graph info
    formatted_dict["log_p_str"] = "[%(levelname)-8s] %(message)s"
    return formatted_dict
