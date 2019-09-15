from typing import Any, List

from yeahml.build.components.activation import _configure_activation
from yeahml.build.components.initializer import _configure_initializer
from yeahml.build.components.regularizer import _configure_regularizer
from yeahml.build.layers.config import return_available_layers
from yeahml.build.layers.other import (
    build_batch_normalization_layer,
    build_embedding_layer,
)
from yeahml.helper import fmt_tensor_info


def build_layer(ltype, opts, l_name, logger, g_logger):

    # TODO: could place a ltype = get_ltype_mapping() here
    LAYER_FUNCTIONS = return_available_layers()
    if ltype in LAYER_FUNCTIONS.keys():
        func = LAYER_FUNCTIONS[ltype]["function"]
        try:
            func_args = LAYER_FUNCTIONS[ltype]["func_args"]
            opts.update(func_args)
        except KeyError:
            pass

        # TODO: name should be set earlier, as an opts?
        logger.debug(f"-> START building: {l_name}")
        if opts:
            # TODO: encapsulate this logic, expand as needed
            # could also implement a check upfront to see if the option is valid
            # functions to configure
            for o in opts:
                try:
                    if (
                        o == "kernel_regularizer"
                        or o == "bias_regularizer"
                        or o == "activity_regularizer"
                    ):
                        opts[o] = _configure_regularizer(opts[o])
                    elif o == "activation":
                        opts[o] = _configure_activation(opts[o])
                    elif o == "kernel_initializer" or o == "bias_initializer":
                        opts[o] = _configure_initializer(opts[o])
                except ValueError as e:
                    raise ValueError(
                        f"error creating option {o} for layer {l_name}:\n > {e}"
                    )
                except TypeError as e:
                    raise TypeError(
                        f"error creating option {o} for layer {l_name}:\n > {e}"
                    )
            cur_layer = func(**opts, name=l_name)
        else:
            cur_layer = func(name=l_name)
        g_logger.info(f"{fmt_tensor_info(cur_layer)}")

    return cur_layer


def build_hidden_block(MCd: dict, HCd: dict, logger, g_logger) -> List[Any]:

    logger.info("-> START building hidden block")
    HIDDEN_LAYERS = []

    # build each layer based on the (ordered) yaml specification
    logger.debug(f"loop+start building layers: {HCd['layers'].keys()}")
    for i, l_name in enumerate(HCd["layers"]):
        layer_info = HCd["layers"][str(l_name)]
        opts = layer_info["options"]
        ltype = layer_info["type"].lower()
        logger.debug(f"-> START building: {l_name} ({ltype}) opts: {layer_info}")
        cur_layer = build_layer(ltype, opts, l_name, logger, g_logger)
        logger.debug(f"[End] building: {cur_layer}")

        # elif ltype == "embedding":
        #     cur_layer = build_embedding_layer(opts, l_name, logger, g_logger)
        # elif ltype == "batch_normalization":
        #     cur_layer = build_batch_normalization_layer(opts, l_name, logger, g_logger)
        # elif ltype == "recurrent":
        #     cur_layer = build_recurrent_layer(opts, actfn, l_name, logger, g_logger)
        # else:
        #     raise NotImplementedError(f"layer type: {ltype} not implemented yet")

        HIDDEN_LAYERS.append(cur_layer)

    logger.info("[END] building hidden block")

    return HIDDEN_LAYERS
