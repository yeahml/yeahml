import logging
import os
import sys

# this may be a "dumb" way of accomplishing this, but I want to wait until
# the modules are implemented correctly
def config_logger(MCd: dict, log_type: str):
    def get_level(level_str: str):
        level = None
        if level_str == "DEBUG":
            level = logging.DEBUG
        elif level_str == "INFO":
            level = logging.INFO
        elif level_str == "WARNING":
            level = logging.WARNING
        elif level_str == "ERROR":
            level = logging.ERROR
        elif level_str == "CRITICAL":
            level = logging.CRITICAL
        else:
            level = None

        return level

        # formatting

    c_fmt = logging.Formatter(MCd["log_c_str"])
    f_fmt = logging.Formatter(MCd["log_f_str"])

    if log_type == "build":
        build_logger = logging.getLogger("build_logger")
        build_logger.setLevel(logging.DEBUG)  # set base to lowest level
        b_ch = logging.StreamHandler()
        b_fh = logging.FileHandler(os.path.join(MCd["log_dir"], "yf_logs", "build.log"))

        b_ch.setLevel(get_level(MCd["log_c_lvl"]))
        b_fh.setLevel(get_level(MCd["log_f_lvl"]))

        b_ch.setFormatter(c_fmt)
        b_fh.setFormatter(f_fmt)

        build_logger.addHandler(b_ch)
        build_logger.addHandler(b_fh)
        return build_logger
    elif log_type == "train":
        train_logger = logging.getLogger("train_logger")
        train_logger.setLevel(logging.DEBUG)  # set base to lowest level
        ch = logging.StreamHandler()
        t_fh = logging.FileHandler(os.path.join(MCd["log_dir"], "yf_logs", "train.log"))

        ch.setLevel(get_level(MCd["log_c_lvl"]))
        t_fh.setLevel(get_level(MCd["log_f_lvl"]))

        ch.setFormatter(c_fmt)
        t_fh.setFormatter(f_fmt)

        train_logger.addHandler(ch)
        train_logger.addHandler(t_fh)
        return train_logger
    elif log_type == "eval":
        eval_logger = logging.getLogger("eval_logger")
        eval_logger.setLevel(logging.DEBUG)  # set base to lowest level
        ch = logging.StreamHandler()
        e_fh = logging.FileHandler(os.path.join(MCd["log_dir"], "yf_logs", "eval.log"))

        ch.setLevel(get_level(MCd["log_c_lvl"]))
        e_fh.setLevel(get_level(MCd["log_f_lvl"]))

        e_fh.setFormatter(f_fmt)
        ch.setFormatter(c_fmt)

        eval_logger.addHandler(ch)
        eval_logger.addHandler(e_fh)
        return eval_logger
    else:
        sys.exit("can't get this logger type: {}".format(log_type))

