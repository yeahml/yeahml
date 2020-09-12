import inspect

import tensorflow as tf


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

    # NOTE: currently constraints don't have a "from_config()" method
    # cur_config = constraint_fn().get_config()
    # if func_opt:
    #     if not set(func_opt.keys()).issubset(constraint_obj["func_args"]):
    #         raise ValueError(
    #             f"options {func_opt.keys()} not in {constraint_obj['func_args']}"
    #         )
    #     for k, v in func_opt.items():
    #         cur_config[k] = v
    #     out = constraint_fn().from_config(cur_config)
    # else:
    #     out = constraint_fn()

    return out


def return_available_constraints():

    CONSTRAINT_FUNCTIONS = {}
    available_keras_constraints = tf.keras.constraints.__dict__
    for opt_name, opt_func in available_keras_constraints.items():
        if inspect.isclass(opt_func) and issubclass(
            opt_func, tf.keras.constraints.Constraint
        ):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                CONSTRAINT_FUNCTIONS[opt_name.lower()] = {}
                CONSTRAINT_FUNCTIONS[opt_name.lower()]["function"] = opt_func
                args = inspect.signature(opt_func).parameters
                args = list(args)
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
