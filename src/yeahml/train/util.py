def get_losses_to_update(loss_conf, ds_split):
    tf_train_loss_descs_to_update = []
    for _, desc_dict in loss_conf["track"][ds_split].items():
        #  _ = loss_name
        for _, desc_tf_obj in desc_dict.items():
            # _ = desc_name
            tf_train_loss_descs_to_update.append(desc_tf_obj)
    return tf_train_loss_descs_to_update


def get_next_batch(ds_iter):

    # TODO: this should accept the ds_dict and ds_iter so that if we reach the
    # we can recreate the ds_iter -- this will allow us to keep track of number
    # of passes for each dataset. When we do this, we'll also have to drop the
    # `convert_to_single_pass_iterator` function --- it will be included here
    # instead.
    try:
        batch = next(ds_iter)
    except StopIteration:
        batch = None
    return batch


def re_init_iter(ds_name, split_name, ds_dict):
    return _convert_to_iter(ds_dict[ds_name][split_name])


def _convert_to_iter(tf_ds):
    return tf_ds.repeat(1).__iter__()


def convert_to_single_pass_iterator(ds_dict):
    iter_dict = {}
    for ds_name, ds_name_conf in ds_dict.items():
        iter_dict[ds_name] = {}
        for split_name, tf_ds in ds_name_conf.items():
            # only loop once
            iter_dict[ds_name][split_name] = _convert_to_iter(tf_ds)
    return iter_dict
