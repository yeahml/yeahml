import yeahml as yml


def test_return_available_layers():
    o = yml.build.layers.config.return_available_layers()
    keys = list(o.keys())
    assert len(keys) > 0
    assert set(
        ["conv1d", "conv2d", "dense", "conv3d", "avgpool2d", "dropout"]
    ).issubset(set(keys))
    assert isinstance(o, dict)
    for k in keys:
        assert isinstance(k, str)
