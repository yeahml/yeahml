import tensorflow_datasets as tfds


def obtain_datasets(dataset_name="", splits=[]):

    # TODO: another option should be to use the common/listed test split
    # TODO: ensure:
    # 1) shuffling
    # 2) consistency

    # https://github.com/tensorflow/datasets/blob/e53f3af5997bd0af9f7e61de3b8c98d8254e07b6/docs/splits.md
    if not dataset_name:
        raise ValueError(
            f"no dataset specified. please select one of {tfds.list_builders()}"
        )

    if not splits:
        raise ValueError(
            f"no split specified. Please specify a split. an example may be {[75, 15, 10]}"
        )
    # TODO: ensure this:
    # 1) shuffles
    # 2) is consistent each call
    train_split, valid_split, test_split = tfds.Split.TRAIN.subsplit(splits)
    train_set = tfds.load(dataset_name, split=train_split, as_supervised=True)
    valid_set = tfds.load(dataset_name, split=valid_split, as_supervised=True)
    test_set = tfds.load(dataset_name, split=test_split, as_supervised=True)

    return train_set, valid_set, test_set
