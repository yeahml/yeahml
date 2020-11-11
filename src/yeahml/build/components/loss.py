import inspect

import tensorflow as tf

from yeahml.build.components.util import copy_func


def return_available_losses():

    LOSS_FUNCTIONS = {}
    available_keras_losses = tf.losses.__dict__

    for opt_name, opt_func in available_keras_losses.items():
        if callable(opt_func) and not inspect.isclass(opt_func):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                LOSS_FUNCTIONS[opt_name.lower()] = {}
                LOSS_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = inspect.signature(opt_func).parameters
                args = [a for a in args if a not in ["y_pred", "y_true"]]
                LOSS_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return LOSS_FUNCTIONS


def return_loss(loss_str):
    avail_losses = return_available_losses()
    try:
        loss = avail_losses[loss_str]
    except KeyError:
        raise KeyError(
            f"loss {loss_str} not available in options {avail_losses.keys()}"
        )

    return loss


def configure_loss(opt_dict):
    """expects :type and :options"""

    try:
        cur_type = opt_dict["type"]
    except TypeError:
        # TODO: could include more helpful message here if the type is an initializer option
        raise TypeError(
            f"config for loss does not specify a 'type'. Current specified options:({opt_dict})."
        )
    loss_obj = return_loss(cur_type.lower())
    loss_fn = loss_obj["function"]

    try:
        options = opt_dict["options"]
        if options:
            if isinstance(options, list):
                if len(options) > 1:
                    raise ValueError(
                        f"there are more than one options dict specified for for {opt_dict}"
                    )
                options = options[0]
            else:
                raise ValueError(
                    f"expected options to be of type list, not {type(options)} -- options: {options}"
                )

    except KeyError:
        options = None

    if options:
        if not set(options.keys()).issubset(loss_obj["func_args"]):
            raise ValueError(f"options {options.keys()} not in {loss_obj['func_args']}")

        # in some cases the loss function .__dict__ does not return any function
        # values (when it should) as a result of being wrapped, this block will
        # obtain the original function
        if "__original_wrapped__" in loss_fn.__dict__:
            loss_fn = copy_func(loss_fn.__dict__["__original_wrapped__"])
        else:
            loss_fn = copy_func(loss_fn)

        var_list = list(loss_fn.__code__.co_varnames)
        # TODO: there must be a more `automatic` way to filter these
        var_list = [
            e
            for e in var_list
            if e not in ("y_pred", "y_true") and not e.startswith("_")
        ]

        cur_defaults_list = list(loss_fn.__defaults__)
        for ao, v in options.items():
            arg_index = var_list.index(ao)
            # TODO: same type assertion?
            cur_defaults_list[arg_index] = v
        loss_fn.__defaults__ = tuple(cur_defaults_list)

    return loss_fn
