def configure_pooling_layer(hld: dict) -> dict:

    try:
        if opts:
            pool_size = opts["pool_size"]
        else:
            pool_size = 3
    except KeyError:
        pool_size = 3
    logger.debug("pool_size set: {}".format(pool_size))
