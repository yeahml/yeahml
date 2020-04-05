from yeahml.dataset.util import get_configured_dataset


def get_datasets(datasets, data_cdict, hp_cdict):

    # TODO: this section is going to be rewritten to match the dataduit config.
    datasets_dict = {}

    # TODO: there likely needs to be different configurations for each datasets
    # (augmentations etc.)

    # TODO: there needs to be some check here to ensure the same datsets are being compared.
    if not datasets:
        raise NotImplementedError(
            "this section needs to be rewritten to match the dataduit api"
        )
        train_ds = get_configured_dataset(
            data_cdict, hp_cdict, ds=None, ds_type="train"
        )
        val_ds = get_configured_dataset(data_cdict, hp_cdict, ds=None, ds_type="val")
    else:
        # TODO: apply shuffle/aug/reshape from config
        # TODO: this is a bandaid fix. this data api is very rough
        for dataset_name, dataset_config in data_cdict["datasets"].items():
            datasets_dict[dataset_name] = {}
            splits = dataset_config["split"]["names"]
            for data_split_name in splits:
                try:
                    tf_ds_raw = datasets[dataset_name][data_split_name]
                except KeyError:
                    raise KeyError(
                        f"The datasets included do not contain {dataset_name}:{data_split_name} -- datasets:{datasets}"
                    )
                tf_ds = get_configured_dataset(data_cdict, hp_cdict, ds=tf_ds_raw)
                datasets_dict[dataset_name][data_split_name] = tf_ds

    return datasets_dict
