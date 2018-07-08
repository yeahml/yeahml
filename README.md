# YamlFlow

[//]: # (Image References)
[tensorboard_graph]: ./misc/tensorboard_graph.png
[tensorboard_scalar]: ./misc/tensorboard_scalar.png

YamlFlow is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).

The goal of the core implementation is as follows:

Where documentation+examples for the main configuration file can be found [here](./documentation_helper/configuration_files/model_config.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./documentation_helper/configuration_files/hidden_config.md). Additional information, such as documentation for the currently supported layer types [conv2d](./documentation_helper/configuration_files/layers/conv2d.md), [pooling](./documentation_helper/configuration_files/layers/pooling2d.md), and [dense](./documentation_helper/configuration_files/layers/dense.md) are also included.

## Main use

```python
import yamlflow as yf

## parse config files
model_config, hidden_config = yf.create_model_and_hidden_config(
    "./example/cats_v_dogs_01/model_config.yaml"
)

## build graph
g = yf.build_graph(model_config, hidden_config)

## train graph
_ = yf.train_graph(g, model_config)

## evaluate graph
_ = yf.eval_graph(g, model_config)
```

### Configuration Files

The model config may look similar to the following:

```yaml
overall:
  name: 'mnist'
  type: 'softmax'
  experiment_dir: 'trial_01'
  saver:
    save_params_name: "best_params_saver"
    load_params_path: "./example/mnist/saved_params/best_params/best_params_ckpt.ckpt" # default location to load parameters from for transfer learning
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
    dir: './example/mnist/data/'
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
  yaml: './example/mnist/hidden_config.yaml'
#train:
  #image_standardize: True
  #augmentation:
    #aug_val: True
    #v_flip: True
    #h_flip: True
```

The hidden layer architecture config (where the path to this file is specified above by (`hidden:yaml`) may look similiar to the following:

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
  - will output information about building the graph
- `train.log`
  - will output information about training the graph
- `graph.log`
  - will output human readable, basic, information about the graph
- `eval.log`
  - will output information about evaluating the graph

## Getting Started

There is currently one example project [here](./example/cats_v_dogs_01/README.md) that the main files implement.

Main documentation + getting started will be created after a logical project structure is created and more examples have been tested (currently only one, cats vs dogs binary image classification). **If anyone would like to attempt to use or work on this, feel free to open an issue and/or reach out to me on twitter @Jack_Burdick**

## Motivation

The primary motivation is to define and create (simple) models easily (for machines and humans). The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a seperate project (TODO: inlude link once made) that will attempt to generate/analyze architectures.

### Future Goals

At the moment the project is being developed around a binary image classification task. In the future, I'd like to support:

- Regression
- Autoencoders
- GANs
- Segmentation

### TODO and in Progress

- handle instance id when not present in the dasaset (mnist) - include warning
- include 'WIPE' option to delete current logs as needed
- support additional metrics
- regression support (use Cali Housing)
- include documentation on what is possible (likely when the project is a little further along)
- don't wipe directories when running eval
- handle class imbalance (this is a bit loaded...)
- find way to simplify/standarize input type (beyond the current three tfrecords)
- find way to automate the reshaping of the label from the batch
- output preds csv (with format string) -- could be useful for competitions
- load params from specified paths for specified layers (beyond the default file)
  - this could be, potentially, useful for concatenation layer
- basic sanity check for building the parse/config file
- support type and name declarations from config for parsing tf records
- logging for initializing weights
  - remove FULL_ERROR
- resource management/device placement
- config option for one_hot -- currently requires manual input
- An option for displaying sample outputs during training/evaluation
- allow prediction from regular values
  - not just tfrecords. This will also be important for serving implications
- Binary classification with the IMDB dataset
- Support `opts` for the optimizer
- Support different types of loss functions (right now these are hardcoded by (type)

### TODO: stretch

- An option for implementing ~smooth grad + for visualizations/ interpret-ability
- capsule layers (and scalar -> vect, vect -> scalar?)
- methodolgy for reducing model size for serving - plot performance as the dtype of operations are reduced / can some operations be removed from the graph?
- support k fold cross validation
- prepackaged models + params trained on well known datasets

#### Notes

Why YAML?

> Simply, because I wanted support for comments. Though x,y,or z may be ``better'' configuration types, yaml was selected for its general popularity and support for comments. I'm open to changing this in the future -- doing so would only require parsing the new config standard into the expect python dict -- but at the moment it isn't a top priority for me personally.  Additionally, since the interface/functionality will likely change dramatically, I think it would be best to stick with the one config file option (yaml) for now.
