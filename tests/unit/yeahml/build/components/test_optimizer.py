import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("optimizer")


def test_return_available_optimizers():
    """test the return type and existence of available optimizer"""
    o = yml.build.components.optimizer.return_available_optimizers()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("optimizer", available_components, ids=available_components)
def test_common_optimizer_available(optimizer):
    """test that common optimizer are available"""
    o = yml.build.components.optimizer.return_available_optimizers()
    keys = set(o.keys())
    assert optimizer in keys
