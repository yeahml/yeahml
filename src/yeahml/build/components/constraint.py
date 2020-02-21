import inspect

import tensorflow as tf


def _configure_constraint(opt_dict):
    try:
        cur_type = opt_dict["type"]
    except TypeError:
        # TODO: could include more helpful message here if the type is an initializer option
        raise TypeError(
            f"config for initialier does not specify a 'type'. Current specified options:({opt_dict})."
        )
    constraint_obj = return_constraint(cur_type.lower())
    constraint_fn = constraint_obj["function"]

    cur_opts = None
    try:
        cur_opts = opt_dict["options"]
    except KeyError:
        pass
    if cur_opts:
        if not set(cur_opts.keys()).issubset(constraint_obj["func_args"]):
            raise ValueError(
                f"options {opt_dict['options'].keys()} not in {constraint_obj['func_args']}"
            )
        out = constraint_fn(**cur_opts)
    else:
        out = constraint_fn()

    return out


def configure_constraint(func_type, func_opt):
    constraint_obj = return_constraint(func_type)
    constraint_fn = constraint_obj["function"]

    if func_opt:
        if not set(func_opt.keys()).issubset(constraint_obj["func_args"]):
            raise ValueError(
                f"options {func_opt.keys()} not in {constraint_obj['func_args']}"
            )
        out = constraint_fn(**func_opt)
    else:
        out = constraint_fn()

    return out


def return_available_constraints():

    CONSTRAINT_FUNCTIONS = {}
    available_keras_constraints = tf.keras.constraints.__dict__
    for opt_name, opt_func in available_keras_constraints.items():
        if inspect.isclass(opt_func) and issubclass(
            opt_func, tf.keras.constraints.Constraint
        ):  # callable(opt_func):  # or
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                CONSTRAINT_FUNCTIONS[opt_name.lower()] = {}
                CONSTRAINT_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                try:
                    # args = opt_func().get_config()
                    args = list(vars(opt_func)["__init__"].__code__.co_varnames)
                    args = [a for a in args if a != "self"]
                except KeyError:
                    args = None

                CONSTRAINT_FUNCTIONS[opt_name.lower()]["func_args"] = args
    return CONSTRAINT_FUNCTIONS


def return_constraint(constraint_str):
    avail_constraints = return_available_constraints()
    try:
        # NOTE: this feels like the wrong place to add a .lower()
        constraint = avail_constraints[constraint_str.lower()]
    except KeyError:
        raise KeyError(
            f"constraint {constraint_str} not available in options {avail_constraints.keys()}"
        )

    return constraint
