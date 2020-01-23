import json
from pathlib import Path

import pytest

import yeahml as yml

# a common set of layers was written to a pickle file - load these
# NOTE: the root of these common available names may belong in a fixture but I'm
# unsure how to do this at this point.
ROOT_YML_DIR = Path(yml.__file__).parent
CUR_AVAIL = ROOT_YML_DIR.joinpath("config").joinpath("available")
available_layers_path = CUR_AVAIL.joinpath("layers.json")
with open(available_layers_path, "r") as fp:
    data_dict = json.load(fp)
    available_layers = data_dict["data"]


def test_return_available_layers():
    """test the return type and existance of available layers"""
    o = yml.build.layers.config.return_available_layers()
    keys = list(o.keys())
    assert len(keys) > 0
    assert isinstance(o, dict)
    for k in keys:
        assert isinstance(k, str)


@pytest.mark.parametrize("layer", available_layers, ids=available_layers)
def test_common_layers_available(layer):
    """test that common layers are available"""
    o = yml.build.layers.config.return_available_layers()
    keys = set(o.keys())
    assert layer in keys
