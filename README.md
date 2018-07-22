# YamlFlow

[//]: # (Image References)
[tensorboard_graph]: ./misc/tensorboard_graph.png
[tensorboard_scalar]: ./misc/tensorboard_scalar.png

<p align="center">
<img src="./misc/yf_logo_draft_02.png" width="300">
</p>

YamlFlow is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).

The goal of the core implementation is as follows:

## Main use

```python
import yamlflow as yf

## parse config files
model_config, hidden_config = yf.create_model_and_hidden_config(
    "./examples/cats_v_dogs_01/model_config.yaml"
)

## build graph
g = yf.build_graph(model_config, hidden_config)

## train graph
_ = yf.train_graph(g, model_config)

## evaluate graph
_ = yf.eval_graph(g, model_config)
```

Where documentation+examples for the main configuration file can be found [here](./docs/configuration_files/model_config.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./docs/configuration_files/hidden_config.md). Additional information, such as documentation for the currently supported layer types [conv2d](./docs/configuration_files/layers/conv2d.md), [pooling](./docs/configuration_files/layers/pooling2d.md), and [dense](./docs/configuration_files/layers/dense.md) are also included.

### Configuration Files

The model config may look similar to the following:

```yaml
overall:
  name: 'mnist'
  loss_fn: 'softmax'
  experiment_dir: 'trial_01'
  saver:
    save_params_name: "best_params_saver"
    load_params_path: "./examples/mnist/saved_params/best_params/best_params_ckpt.ckpt" # default location to load parameters from for transfer learning
  trace: 'full'
  logging:
    console:
      level: 'critical'
      format_str: null
    file:
      level: 'debug'
      format_str: null
    graph_spec: True
data:
  in:
    dim: [784]
    dtype: 'float32'
    reshape_to: [28, 28, 1]
  label:
    dim: [10]
    dtype: 'float32'
  TFR:
    dir: './examples/mnist/data/'
    train: 'train.tfrecords'
    validation: 'validation.tfrecords'
    test: 'test.tfrecords'
hyper_parameters:
  lr: 0.00001
  batch_size: 16
  epochs: 20
  optimizer: 'adam'
  default_activation: 'elu'
  shuffle_buffer: 128
  early_stopping:
    epochs: 3
    warm_up_epochs: 5
hidden:
  yaml: './examples/mnist/hidden_config.yaml'
#train:
  #image_standardize: True
  #augmentation:
    #aug_val: True
    #v_flip: True
    #h_flip: True
```

The hidden layer architecture config (where the path to this file is specified above by (`hidden:yaml`) may look similar to the following:

```yaml
layers:
  conv_1:
    type: 'conv2d'
    options:
      filters: 32
      kernel_size: 3
      strides: 1
      trainable: False # default is True
    saver:
      load_params: True # default is False
  conv_2:
    type: 'conv2d'
    options:
      filters: 64
      kernel_size: 3
      strides: 1
    saver:
      load_params: True
  pool_1:
    type: 'pooling2d'
  dense_1:
    type: 'dense'
    options:
      units: 16
      dropout: 0.5
    saver:
      load_params: True
```

### TensorBoard

After training, tensorboard can be used to inspect the graph and metrics by issuing the following command: `tensorboard --logdir "tf_logs/"` which will open tensorboard and display figures similar to those below.

![Example of TensorFlow graph in tensorboard -- showing the graph and compute time trace][tensorboard_graph]
Where the image on the left is of the specified graph (in the architecture config yaml file) and the graph on the right is showing a trace of compute time of the operations during training (set by the `overall:trace`) parameter in the model config file.

![Example of TensorFlow metrics in tensorboard -- showing scalars and histogram][tensorboard_scalar]

Where the image on the right shows the training and validation metrics during training (computed over the entire iteration) and the right shows histograms of the parameters (weights+biases) calculated during training.

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

## Getting Started

There is currently two example projects [cats vs dogs (binary classification)](./examples/cats_v_dogs_01/) and [mnist (multi-class classification)](./examples/mnist/) that the `example.py` file implements.

Main documentation is ongoing+work in progress. Please reach out if you have questions/concerns.

**If anyone would like to attempt to use or modify this project, feel free to open an issue and/or reach out to me on twitter [@Jack_Burdick](https://twitter.com/Jack_Burdick)**

## Motivation

The primary motivation is to define and create (simple) models easily (for machines and humans). The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a separate project (TODO: include link once made) that will attempt to generate/analyze architectures.

### Future Goals

At the moment the project is being developed around a binary image classification task. In the future, I'd like to support:

1. Regression
2. Segmentation
3. Autoencoders
4. GANs

### TODO and in Progress

- handle instance id when not present in the dataset (mnist) - include warning
- include 'WIPE' option to delete current logs as needed
- support additional metrics
- include documentation on what is possible (likely when the project is a little further along)
- don't wipe directories when running eval
- handle class imbalance (this is a bit loaded...)
- find way to simplify/standardize input type (beyond the current three tfrecords)
- find way to automate the reshaping of the label from the batch
- output preds csv (with format string) -- could be useful for competitions
- load params from specified paths for specified layers (beyond the default file)
  - this could be, potentially, useful for concatenation layer
- basic sanity check for building the parse/config file
- support type and name declarations from config for parsing tf records
- logging for initializing weights
- resource management/device placement
- config option for one_hot -- currently requires manual input
- An option for displaying sample outputs during training/evaluation
- allow prediction from regular values
  - not just tfrecords. This will also be important for serving implications
- Binary classification with the IMDB dataset
- Support `opts` for the optimizer
- Support different types of loss functions (right now these are hardcoded by (type)
- [sphinx](http://www.sphinx-doc.org/en/master/) documentation
- Add docstrings to each function
- Batch normalization layer
- Support for concatenation (would allow for the creation of custom modules -- similar to inception)
- Depthwise separable convolutions
- support k fold cross validation
- make config mapping dictionary read only [related SO](https://stackoverflow.com/questions/19022868/how-to-make-dictionary-read-only-in-python)
- update graph printing/logging methodology -- more table like
- add confusion matrix for classification problems

### TODO: stretch

- An option for implementing ~smooth grad + for visualizations/ interpret-ability
- capsule layers (and scalar -> vect, vect -> scalar?)
- methodology for reducing model size for serving - plot performance as the dtype of operations are reduced / can some operations be removed from the graph?
- prepackaged models + params trained on well known datasets
- support custom loss functions (likely through link to .py file?)
- Break from single sequential pattern
  - Support for multi-input
  - Support for multi-output
- Hyperparameter optimization (hyperopt support?)

#### Notes

Why YAML?

> Simply, because I wanted support for comments. Though x,y,or z may be ``better'' configuration types, yaml was selected for its general popularity and support for comments. I'm open to changing this in the future -- doing so would only require parsing the new config standard into the expect python dict -- but at the moment it isn't a top priority for me personally.  Additionally, since the interface/functionality will likely change dramatically, I think it would be best to stick with the one config file option (yaml) for now.
