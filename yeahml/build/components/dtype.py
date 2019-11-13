import tensorflow as tf


def return_available_dtypes():
    DTYPES = {}
    available_dtypes = tf.dtypes.__dict__
    for opt_name, opt_func in available_dtypes.items():
        if issubclass(type(opt_func), tf.dtypes.DType):
            if opt_name.lower() not in set(["deserialize", "get", "serialize"]):
                DTYPES[opt_name.lower()] = {}
                DTYPES[opt_name.lower()] = opt_func

    return DTYPES


def return_dtype(dtype_str: str):
    avail_dtype = return_available_dtypes()
    try:
        # NOTE: this feels like the wrong place to add a .lower()
        dtype = avail_dtype[dtype_str.lower()]
    except KeyError:
        raise KeyError(
            f"dtype {dtype_str} not available in options {avail_dtype.keys()}"
        )

    return dtype
