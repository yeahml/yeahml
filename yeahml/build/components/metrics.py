import tensorflow as tf
from typing import Any
import inspect


def return_available_metrics():

    METRICS_FUNCTIONS = {}
    available_keras_metrics = tf.keras.metrics.__dict__
    for opt_name, opt_func in available_keras_metrics.items():
        if inspect.isclass(opt_func) and issubclass(opt_func, tf.keras.metrics.Metric):
            if opt_name.lower() not in set(
                ["deserialize", "get", "serialize", "metric"]
            ):
                METRICS_FUNCTIONS[opt_name.lower()] = {}
                METRICS_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                try:
                    args = list(vars(opt_func)["__init__"].__code__.co_varnames)
                    args = [a for a in args if a != "self"]
                except KeyError:
                    args = None

                METRICS_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return METRICS_FUNCTIONS


def return_metric(metric_str):
    avail_metrics = return_available_metrics()
    try:
        metric = avail_metrics[metric_str]
    except KeyError:
        raise KeyError(
            f"metric {metric_str} not available in options {avail_metrics.keys()}"
        )

    return metric


def configure_metric(cur_type, met_opts):

    metric_obj = return_metric(cur_type.lower())
    metric_fn = metric_obj["function"]
    options = met_opts

    if options:
        if not set(opt_dict["options"].keys()).issubset(metric_obj["func_args"]):
            raise ValueError(
                f"options {opt_dict['options'].keys()} not in {init_obj['func_args']}"
            )
        metric_fn = copy_func(metric_fn)
        var_list = list(metric_fn.__code__.co_varnames)
        print(var_list)
        sys.exit()
        cur_defaults_list = list(metric_fn.__defaults__)
        for ao, v in options.items():
            arg_index = var_list.index(ao)
            # TODO: same type assertion?
            cur_defaults_list[arg_index] = v
        metric_fn.__defaults__ = tuple(cur_defaults_list)

    return metric_fn()
