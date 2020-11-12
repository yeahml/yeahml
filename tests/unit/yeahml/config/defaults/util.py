import crummycm as ccm
from yeahml.config.create_configs import MODEL_IGNORE_HASH_KEYS, make_hash
from yeahml.config.default.types.compound.layer import layers_parser
from yeahml.config.default.types.compound.performance import performances_parser

# from yeahml.config.default.types.compound.callbacks import callbacks_parser


def parse_default(user, temp):
    user_pass = user.copy()

    # parse + validate
    config_dict = ccm.validate(user_pass, temp)

    # custom parsers

    if "performance" in user.keys():
        config_dict["performance"]["objectives"] = performances_parser()(
            config_dict["performance"]["objectives"]
        )
    # if "callbacks" in user.keys():
    #     config_dict["callbacks"]["objects"] = callbacks_parser()(
    #         config_dict["callbacks"]["objects"]
    #     )
    if "model" in user.keys():
        config_dict["model"]["layers"] = layers_parser()(config_dict["model"]["layers"])
        model_hash = make_hash(config_dict["model"], MODEL_IGNORE_HASH_KEYS)
        config_dict["model"]["model_hash"] = model_hash

    return config_dict
