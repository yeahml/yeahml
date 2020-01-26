import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("initializer")


def test_return_available_initializers():
    """test the return type and existence of available initializer"""
    o = yml.build.components.initializer.return_available_initializers()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("initializer", available_components, ids=available_components)
def test_common_initializer_available(initializer):
    """test that common initializer are available"""
    o = yml.build.components.initializer.return_available_initializers()
    keys = set(o.keys())
    assert initializer in keys
