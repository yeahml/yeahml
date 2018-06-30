# Architecture Options

The architecture configuration file will contain the configuration information for how the hidden layers in the model described by the main `model_config.yaml` yaml file should be constructed. An example configuration file can be found [here](./examples/architecture.yaml).

## Overview

The hidden layer architecture can be defined by a `.yaml` with the following basic guidelines. The `.yaml` file begins with the `layers` keyword and each subsequent entry defines a layer.

Each layer should follow a similar and predictable pattern, where the top most value `<name>` will be the name of the layer. The type of the layer will be defined next in `type: <type of layer>` (see below for the currently supported types of layers). The options field will then contain any options (that are specific to the `type` of layer being constructed).

Note that comments are allowed.

```yaml
layers:
  <name>:
    type: <type of layer>
    options:
      <option name>: <option value>
```

## Currently Supported Layer Types

NOTE: only 2d types are currently supported i.e. conv2d is supported by conv3d is not (yet).

- convolution
  - implements: [tf.layers.conv2d](https://www.tensorflow.org/api_docs/python/tf/layers/conv2d)
  - information: [./layers/conv2d](./layers/conv2d.md)
- pooling
  - implements: [tf.layers.max_pooling2d](https://www.tensorflow.org/api_docs/python/tf/layers/max_pooling2d)
  - information: [./layers/conv2d](./layers/pooling2d.md)
- dense
  - implements: [tf.layers.dense](https://www.tensorflow.org/api_docs/python/tf/layers/dense)
  - information: [./layers/conv2d](./layers/dense.md)

---

## Connecting Layers

There is no need to specify input type/shape. The model will be created according to the order you specify and the appropriate connections will be made for you*

```yaml
layers:
  conv_3:
    type: 'conv2d'
    options:
      filters: 4
      kernel_size: 3
      strides: 2
  pool_1:
    type: 'pooling2d'
  dense_1:
    type: 'dense'
    options:
      units: 64
      dropout: 0.5
```
*this is still a beta/prototype project, please understand that not everything may be connected correctly.

## Shape information

When a graph is created the layer shape information will be printed to the terminal. For example,

```bash
========================cats vs dogs========================
| inputs/X_in     | (?, 150, 150, 3)
| conv_1/Elu      | (?, 75, 75, 16)
| conv_2/Elu      | (?, 38, 38, 32)
| conv_3/Elu      | (?, 19, 19, 64)
| pool_1/MaxPool  | (?, 9, 9, 64)
>> flatten: (?, 5184)
| dense_1/Elu     | (?, 64)
>> dropout: 0.5
| dense_2/Elu     | (?, 16)
>> dropout: 0.5
| preds           | (?, 1)
opt: Adam
============================================================

```


### Other Notes

Currently, only simple architectures, that follow common patterns, can be defined. When creating a CNN architecture, it is not necessary to flatten the pooling/convolutional layer before the dense layer --- this will be managed behind the scenes with a `flatten` operation.