import numpy as np


def parse_overall(MC: dict) -> dict:
    MCd = {}
    MCd["name"] = MC["overall"]["name"]
    MCd["experiment_dir"] = MC["overall"]["experiment_dir"]
    try:
        MCd["seed"] = MC["overall"]["rand_seed"]
    except KeyError:
        pass

    try:
        MCd["trace_level"] = MC["overall"]["trace"].lower()
    except KeyError:
        pass

    # TODO: this is a temp+new object in the dict
    try:
        MCd["num_classes"] = MC["overall"]["num_classes"]
    except KeyError:
        # no params will be loaded from previously trained params
        # TODO: I don't feel great about this.. this is a temp fix
        if MC["overall"]["metrics"]["type"] == "regression":
            MCd["num_classes"] = 1
        pass

    if (
        MC["overall"]["metrics"]["type"] == "classification"
        or MC["overall"]["metrics"]["type"] == "segmentation"
    ):
        try:
            MCd["class_weights"] = np.asarray(MC["overall"]["class_weights"])
        except KeyError:
            MCd["class_weights"] = np.asarray([1.0] * MCd["num_classes"])
    return MCd
