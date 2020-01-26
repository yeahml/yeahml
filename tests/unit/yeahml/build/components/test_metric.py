import pytest

import yeahml as yml
from util import return_data_and_ids

available_components = return_data_and_ids("metric")


def test_return_available_metrics():
    """test the return type and existence of available metric"""
    o = yml.build.components.metric.return_available_metrics()
    keys = list(o.keys())
    assert len(keys) > 0, f"there are no available components found"
    assert isinstance(o, dict), f"object is not a dictionary"
    for k in keys:
        assert isinstance(k, str), f"the key {k} is not a string (type: {type(k)})"


@pytest.mark.parametrize("metric", available_components, ids=available_components)
def test_common_metric_available(metric):
    """test that common metric are available"""
    o = yml.build.components.metric.return_available_metrics()
    keys = set(o.keys())
    assert metric in keys
