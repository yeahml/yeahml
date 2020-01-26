import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("dtype")


def test_return_available_dtypes():
    """test the return type and existence of available dtype"""
    o = yml.build.components.dtype.return_available_dtypes()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("dtype", available_components, ids=available_components)
def test_common_dtype_available(dtype):
    """test that common dtype are available"""
    o = yml.build.components.dtype.return_available_dtypes()
    keys = set(o.keys())
    assert dtype in keys
