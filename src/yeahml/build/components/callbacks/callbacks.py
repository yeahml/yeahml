import inspect
import importlib
from pathlib import Path


import tensorflow as tf


def configure_callback(callback_dict):
    # TODO: this should not be here. (follow template for losses)
    callback_conf = return_callback(callback_dict["type"])
    callback_fn = callback_conf["function"]

    # configure callback
    temp_dict = callback_dict.copy()
    if temp_dict["options"]:
        callback = callback_fn(**temp_dict["options"])
    else:
        callback = callback_fn()

    return callback


def return_available_callbacks():
    # logic to get all layers in a class
    CALLBACK_FUNCTIONS = {}
    available_keras_callbacks = tf.keras.callbacks.__dict__
    for cb_name, cb_func in available_keras_callbacks.items():
        # TODO: could change to is subclass tf.keras.optimizers.Optimizer
        if cb_name.lower() != "callback":  # NOTE: hardcoded
            if inspect.isclass(cb_func):
                CALLBACK_FUNCTIONS[cb_name.lower()] = {}
                CALLBACK_FUNCTIONS[cb_name.lower()]["function"] = cb_func

                # I'm not sure why, but the `terminateonnan` is different than
                # the others.. I haven't looked into it as this skirts the problem
                if inspect.signature(cb_func).parameters:
                    args = inspect.signature(cb_func.__dict__["__init__"]).parameters
                    args = [a for a in args if a not in ["self", "kwargs"]]
                else:
                    args = []

                filt_args = [a for a in args if a not in ["self", "kwargs"]]
                CALLBACK_FUNCTIONS[cb_name.lower()]["func_args"] = filt_args

    custom_callbacks = {}
    # TODO: I would imagine there is a better way to do this
    cb_cb_dir = Path(__file__).parent.joinpath("objects")
    files = [
        x
        for x in cb_cb_dir.iterdir()
        if x.is_file() and str(x).split("/")[-1].split(".")[0] not in ["base"]
    ]
    for import_source in files:
        import_source = str(import_source).split("/")
        import_source = import_source[import_source.index("yeahml") :]
        import_source = ".".join(import_source)
        if import_source.endswith(".py"):
            import_source = import_source.rstrip("py").rstrip(".")
        custom_mod = importlib.import_module(f"{import_source}")
        for cb_name, cb_class in inspect.getmembers(custom_mod, inspect.isclass):
            if inspect.isclass(cb_class):
                custom_callbacks[cb_name.lower()] = {}
                custom_callbacks[cb_name.lower()]["function"] = cb_class

                if inspect.signature(cb_class).parameters:
                    # args = inspect.signature(cb_class.__dict__["__init__"]).parameters
                    args = inspect.signature(cb_class).parameters
                    args = [a for a in args if a not in ["self", "kwargs"]]
                else:
                    args = []

                filt_args = [a for a in args if a not in ["self", "kwargs"]]
                custom_callbacks[cb_name.lower()]["func_args"] = filt_args

    # TODO: in the future, we may either override or wrap the existing callback
    # in a class that profices the information w/o relying on Keras's way of
    # doing things e.g. self.optimizer
    tf_cb_keys = CALLBACK_FUNCTIONS.keys()
    for cust_cb in list(custom_callbacks.keys()):
        if cust_cb in tf_cb_keys:
            raise ValueError(
                f"custom callback {cust_cb} will override standard callback"
            )
    CALLBACK_FUNCTIONS = {**CALLBACK_FUNCTIONS, **custom_callbacks}

    return CALLBACK_FUNCTIONS


def return_callback(callback_str):
    avail_callbacks = return_available_callbacks()
    try:
        callback = avail_callbacks[callback_str]
    except KeyError:
        raise KeyError(
            f"callback {callback_str} not available in options {avail_callbacks.keys()}"
        )

    return callback

