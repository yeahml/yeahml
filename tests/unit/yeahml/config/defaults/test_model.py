import pytest

from yeahml.config.default.create_default import DEFAULT_CONFIG
from yeahml.config.model.parse_model import format_model_config
import tensorflow as tf

"""
layers:
  dense_1:
    type: 'dense'
    options:
      units: 16
  dense_2:
    type: 'dense'
    options:
      units: 8
      activation:
        type: 'linear'
  dense_3_output:
    type: 'dense'
    options:
      units: 1
      activation:
        type: 'linear'
"""

# TODO: test options
# NOTE: multiple of the same name would need to be caught before here since the
# type is a dict and the same name will overwrite the current value
ex_config = {
    # ----- REQUIRED
    # missing
    "minimal_00": (
        {
            "model": {
                "layers": {
                    "dense_1": {
                        "layer_type": "dense",
                        "layer_options": {"units": 16},
                        "in_name": "jack",
                    },
                    "bn_1": {
                        "layer_type": "batchnormalization",
                        "layer_options": None,
                        "in_name": "jack",
                    },
                }
            }
        },
        {
            "layers": {
                "dense_1": {
                    "layer_base": {
                        "str": "dense",
                        "func": tf.keras.layers.Dense,
                        "func_args": [
                            "self",
                            "units",
                            "activation",
                            "use_bias",
                            "kernel_initializer",
                            "bias_initializer",
                            "kernel_regularizer",
                            "bias_regularizer",
                            "activity_regularizer",
                            "kernel_constraint",
                            "bias_constraint",
                            "trainable",
                            "name",
                            "dtype",
                            "dynamic",
                        ],
                        "func_defaults": [
                            None,
                            True,
                            "glorot_uniform",
                            "zeros",
                            None,
                            None,
                            None,
                            None,
                            None,
                            True,
                            None,
                            None,
                            False,
                        ],
                    }
                },
                "bn_1": {
                    "layer_base": {
                        "str": "batchnormalization",
                        "func": tf.keras.layers.BatchNormalization,
                        "func_args": [
                            "self",
                            "axis",
                            "momentum",
                            "epsilon",
                            "center",
                            "scale",
                            "beta_initializer",
                            "gamma_initializer",
                            "moving_mean_initializer",
                            "moving_variance_initializer",
                            "beta_regularizer",
                            "gamma_regularizer",
                            "beta_constraint",
                            "gamma_constraint",
                            "renorm",
                            "renorm_clipping",
                            "renorm_momentum",
                            "fused",
                            "trainable",
                            "virtual_batch_size",
                            "adjustment",
                            "name",
                            "trainable",
                            "name",
                            "dtype",
                            "dynamic",
                        ],
                        "func_defaults": [
                            -1,
                            0.99,
                            0.001,
                            True,
                            True,
                            "zeros",
                            "ones",
                            "zeros",
                            "ones",
                            None,
                            None,
                            None,
                            None,
                            False,
                            None,
                            0.99,
                            None,
                            True,
                            None,
                            None,
                            None,
                            True,
                            None,
                            None,
                            False,
                        ],
                    }
                },
            }
        },
    )
}


@pytest.mark.parametrize(
    "config,expected", ex_config.values(), ids=list(ex_config.keys())
)
def test_default(config, expected):
    """test parsing of model"""
    if isinstance(expected, dict):
        formatted_config = format_model_config(config["model"], DEFAULT_CONFIG["model"])
        try:
            assert expected == formatted_config
        except AssertionError:
            for k, d in formatted_config["layers"].items():
                for opt in ["func", "func_args", "func_defaults"]:
                    assert (
                        d["layer_base"][opt] == expected["layers"][k]["layer_base"][opt]
                    ), f"layer {k} does not have matching {opt}"

    elif isinstance(expected, ValueError):
        with pytest.raises(ValueError):
            formatted_config = format_model_config(
                config["model"], DEFAULT_CONFIG["model"]
            )
    elif isinstance(expected, TypeError):
        with pytest.raises(TypeError):
            formatted_config = format_model_config(
                config["model"], DEFAULT_CONFIG["model"]
            )
