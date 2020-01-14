# YeahML

YeahML is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml or json).

Examples are currently being worked through and can be found in the [examples directory](./examples)

The core implementation is as follows:

1. Create `.tfrecords` for a training, validation, and test set (accepting pre-made tf dataset in progress [tfd branch])
2. Write configuration files (json or yaml, see below)
  - for the main implementation (meta data, hyperparameters, etc)
  - for the hidden layers
3. Use python (as shown below) to train and evaluate the model
  - There are three main functions
    - build_model
    - train_model
    - eval_model (will load the "best params" from training before evaluating)
4. Iterate on models and ideas
  - logs are created and can be used for debugging or evaluating models
  - tensorboard is ready to use
    - metric scalars
    - parameter histograms and distributions

## Main use

```python
example = "./examples/mnist/main_config.yaml"
config_dict = yml.create_configs(example)

# build graph
model = yml.build_model(config_dict)

# train graph
train_dict = yml.train_model(model, config_dict)

# evaluate graph -- will load the "best params"
eval_dict = yml.eval_model(model, config_dict)


```

Where documentation+examples for the main configuration file can be found [here](./docs/configuration_files/model_cdict.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./docs/configuration_files/hidden_config.md).

## [Examples](./examples)

The included [notebook](./example_notebook.ipynb) shows basic functionality on mnist. There are included example configuration files on other tasks located in the [./examples](./examples) directory (however, these are still a work in progress) *Note: if you have another example you would like to see feel free to reach out to me or make a pull request.*


### Configuration Files

The model config may look similar to the following:

```yaml
meta:
  name: 'mnist'
  experiment_dir: 'trial_01'
  saver:
    save_params_name: "best_params_saver"

logging:
  console:
    level: 'info'
    format_str: null
  file:
    level: 'ERROR'
    format_str: null
  graph_spec: True

performance:
  loss_fn: 
    type: 'categorical_crossentropy'
  type: ["MeanSquaredError", "TopKCategoricalAccuracy", "CategoricalAccuracy"]
  options: [null, 
            {"k": 2}, 
            null]

data:
  in:
    dim: [784]
    dtype: 'float32'
    reshape_to: [28, 28, 1]
  label:
    dim: [10]
    dtype: 'float32'
    one_hot: True # TODO: ensure this is being used
  TFR_parse:
    feature:
      name: "/image"
      dtype: "int8"
      tftype: "fixedlenfeature"
      in_type: "string"
    label:
      name: "/label"
      dtype: "int64" # what to decode it to
      tftype: "fixedlenfeature"
      in_type: "int64" # what it is encoded as
  TFR:
    dir: './examples/mnist/data/'
    train: 'train.tfrecords'
    validation: 'validation.tfrecords'
    test: 'test.tfrecords'

hyper_parameters:
  optimizer: 
    type: 'adam'
    learning_rate: 0.0001
  batch_size: 32
  epochs: 3
  shuffle_buffer: 128
model:
  path: './examples/mnist/model_config.yaml'
```

The model config (where the path to this file is specified above by (`model:path`) may look similar to the following:

```yaml
meta:
  name: "model_a"
  name_override: False
  activation:
    type: 'elu'

layers:
  conv_1:
    type: 'conv2d'
    options:
      filters: 32
      kernel_size: 3
  conv_2:
    type: 'conv2d'
    options:
      filters: 64
      kernel_size: 3
  dropout_1:
    type: 'dropout'
    options:
      rate: 0.5
  pool_1:
    type: 'AveragePooling2D'
    options:
      pool_size: 2
      strides: 2
  flatten_1:
    type: 'flatten'
  dense_1:
    type: 'dense'
    options:
      units: 16
      kernel_regularizer:
        type: 'L1'
        l: 0.01
  dense_2_output:
    type: 'dense'
    options:
      units: 10
      activation:
        type: 'softmax'
```

### TensorBoard

After training, [tensorboard](https://www.tensorflow.org/tensorboard) can be used to inspect the graph and metrics by issuing the following command: `tensorboard --logdir "tf_logs/"` which will open tensorboard and display figures similar to those below.

### Logging

Logging, if enabled, will produce the following log files:

- `build.log`
  - Contains information about building the graph. [Information + Example](./docs/logs/build.md)
- `eval.log`
  - Contains information about evaluating the graph [Information + Example](./docs/logs/eval.md)
- `graph.log`
  - Contains human readable, basic, information about the graph [Information + Example](./docs/logs/graph.md)
- `preds.log`
  - Contains logs of each evaluation set (label, predicted, ground truth, confidence) [Information + Example](./docs/logs/preds.md)
- `train.log`
  - Contains information about training the graph [Information + Example](./docs/logs/train.md)

## Development

This project and documentation is ongoing+work in progress. Please reach out if you have questions or concerns.

**If anyone would like to attempt to use or modify this project, feel free to open an issue and/or reach out to me on twitter [@Jack_Burdick](https://twitter.com/Jack_Burdick)**

## Motivation

The primary motivation is to define and create models easily. The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a separate project (TODO: include link once made) that will attempt to generate/analyze architectures.


### Other:
If you experience problems with using the tensorboard extension in jupyter, try running this script found [here](https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/tools/diagnose_tensorboard.py)