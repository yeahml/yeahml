import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("regularizer")


def test_return_available_regularizers():
    """test the return type and existence of available regularizer"""
    o = yml.build.components.regularizer.return_available_regularizers()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("regularizer", available_components, ids=available_components)
def test_common_regularizer_available(regularizer):
    """test that common regularizer are available"""
    o = yml.build.components.regularizer.return_available_regularizers()
    keys = set(o.keys())
    assert regularizer in keys
