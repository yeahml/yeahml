import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("constraint")


def test_return_available_constraints():
    """test the return type and existence of available constraint"""
    o = yml.build.components.constraint.return_available_constraints()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("constraint", available_components, ids=available_components)
def test_common_constraint_available(constraint):
    """test that common constraint are available"""
    o = yml.build.components.constraint.return_available_constraints()
    keys = set(o.keys())
    assert constraint in keys
