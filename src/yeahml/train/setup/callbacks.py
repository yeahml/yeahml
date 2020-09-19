from yeahml.build.components.callbacks.callbacks import configure_callback


def get_callbacks(cb_dict):

    # TODO: this section is going to be rewritten to match the dataduit config.
    cb_list = []
    if cb_dict:
        for cb_name, cb_config in cb_dict["objects"].items():
            cur_cb = configure_callback(cb_config)
            cb_list.append(cur_cb)

    return cb_list

