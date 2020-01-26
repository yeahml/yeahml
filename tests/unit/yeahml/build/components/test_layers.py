import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("layer")


def test_return_available_layers():
    """test the return type and existence of available components"""
    o = yml.build.layers.config.return_available_layers()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("layer", available_components, ids=available_components)
def test_common_layers_available(layer):
    """test that common layers are available"""
    o = yml.build.layers.config.return_available_layers()
    keys = set(o.keys())
    assert layer in keys

