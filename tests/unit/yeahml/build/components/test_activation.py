import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("activation")


def test_return_available_activations():
    """test the return type and existence of available activation"""
    o = yml.build.components.activation.return_available_activations()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("activation", available_components, ids=available_components)
def test_common_activation_available(activation):
    """test that common activation are available"""
    o = yml.build.components.activation.return_available_activations()
    keys = set(o.keys())
    assert activation in keys
