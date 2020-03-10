import inspect

import tensorflow as tf


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


def configure_metric(cur_type, met_opt_dict):
    # TODO: this needs to be tested
    # TODO: this should mirror the `configure_loss` function

    metric_obj = return_metric(cur_type.lower())
    metric_fn = metric_obj["function"]
    options = met_opt_dict

    if options:
        if not set(options.keys()).issubset(metric_obj["func_args"]):
            raise ValueError(
                f"options {options.keys()} not in {metric_obj['func_args']}"
            )
        var_list = list(vars(metric_fn)["__init__"].__code__.co_varnames)
        new_def_dict = {}
        for ao, v in options.items():
            var_list.index(ao)
            # TODO: same type assertion?
            new_def_dict[ao] = v

        metric_fn = metric_fn(**new_def_dict)
    else:
        metric_fn = metric_fn()

    return metric_fn
