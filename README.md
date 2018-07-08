# YamlFlow

[//]: # (Image References)
[tensorboard_graph]: ./misc/tensorboard_graph.png
[tensorboard_scalar]: ./misc/tensorboard_scalar.png

<p align="center">
<img src="./misc/yf_logo_draft_02.png" width="300">
</p>

YamlFlow is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).

The goal of the core implementation is as follows:

Where documentation+examples for the main configuration file can be found [here](./documentation_helper/configuration_files/model_config.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./documentation_helper/configuration_files/hidden_config.md). Additional information, such as documentation for the currently supported layer types [conv2d](./documentation_helper/configuration_files/layers/conv2d.md), [pooling](./documentation_helper/configuration_files/layers/pooling2d.md), and [dense](./documentation_helper/configuration_files/layers/dense.md) are also included.

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

### Configuration Files

The model config may look similar to the following:

```yaml
overall:
  name: 'mnist'
  type: 'softmax'
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
- `eval.log`
  - will output information about evaluating the graph
- `graph.log`
  - will output human readable, basic, information about the graph
- `preds.log`
  - will output logs of each evaluation set (label, predicted, ground truth, confidence)
- `train.log`
  - will output information about training the graph

#### build.log

> log regarding the contruction of the graph

```python
[2018-07-05 18:54:16,018 - build_hidden.py:79 -   build_conv2d_layer()][DEBUG   ]: Final tensor obj: Tensor("conv_1/Elu:0", shape=(?, 75, 75, 16), dtype=float32)
[2018-07-05 18:54:16,019 - build_hidden.py:82 -   build_conv2d_layer()][DEBUG   ]: [End] building: conv_1
[2018-07-05 18:54:16,019 - build_hidden.py:246 -   build_hidden_block()][DEBUG   ]: -> START building layer: pool_1 with opts: {'type': 'pooling2d', 'options': {'pool_type': 'avg'}}
[2018-07-05 18:54:16,019 - build_hidden.py:265 -   build_hidden_block()][DEBUG   ]: activation set: <function elu at 0x7f44466a9d08>
[2018-07-05 18:54:16,019 - build_hidden.py:292 -   build_hidden_block()][DEBUG   ]: -> START building: pooling2d
[2018-07-05 18:54:16,019 - build_hidden.py:163 -     build_pool_layer()][DEBUG   ]: pool_size set: [2, 2]
[2018-07-05 18:54:16,019 - build_hidden.py:172 -     build_pool_layer()][DEBUG   ]: strides set: 2
[2018-07-05 18:54:16,019 - build_hidden.py:176 -     build_pool_layer()][DEBUG   ]: name set: pool_1
[2018-07-05 18:54:16,019 - build_hidden.py:187 -     build_pool_layer()][DEBUG   ]: pool_type set: avg
[2018-07-05 18:54:16,020 - build_hidden.py:201 -     build_pool_layer()][DEBUG   ]: tensor obj pre dropout: Tensor("pool_1/AvgPool:0", shape=(?, 37, 37, 16), dtype=float32)
[2018-07-05 18:54:16,020 - build_hidden.py:212 -     build_pool_layer()][DEBUG   ]: dropout_rate set: None
[2018-07-05 18:54:16,020 - build_hidden.py:226 -     build_pool_layer()][DEBUG   ]: [End] building: pool_1
```

#### eval.log

> logs from evaluating the graph. This log only logs information about contructing/testing, not about each instance

```python
[2018-07-05 18:54:52,593 - eval_graph.py:63 - eval_graph_from_saver()][INFO    ]: eval_graph_from_saver
```

#### graph.log

> log regarding the human readable graph report

```python
graph_logger: INFO     =============GRAPH=============
graph_logger: INFO     | inputs/X_in     | (?, 150, 150, 3)
graph_logger: INFO     | conv_1/Elu      | (?, 75, 75, 16)
graph_logger: INFO     | pool_1/AvgPool  | (?, 37, 37, 16)
graph_logger: INFO     | conv_2/Elu      | (?, 37, 37, 32)
graph_logger: INFO     | pool_2/MaxPool  | (?, 18, 18, 32)
graph_logger: INFO     | conv_3/Elu      | (?, 18, 18, 64)
graph_logger: INFO     | pool_3/MaxPool  | (?, 9, 9, 64)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     >> flatten: (?, 5184)
graph_logger: INFO     | dense_1/Elu     | (?, 64)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     | dense_2/Elu     | (?, 16)
graph_logger: INFO     >> dropout: 0.5
graph_logger: INFO     | y_proba         | (?, 1)
graph_logger: INFO     Adam
graph_logger: INFO     ==============END==============
```

#### preds.log

> log regarding each test instance (label, predicted, ground truth, confidence)

```python
[INFO    ] b'dog.7748.jpg'  : pred: 1, true: 1, conf: 0.76948
[INFO    ] b'dog.7966.jpg'  : pred: 1, true: 1, conf: 0.75441
[INFO    ] b'cat.3052.jpg'  : pred: 0, true: 0, conf: 0.20389
[INFO    ] b'cat.11598.jpg' : pred: 0, true: 0, conf: 0.17646
[INFO    ] b'dog.9056.jpg'  : pred: 1, true: 1, conf: 0.66992
[INFO    ] b'cat.10145.jpg' : pred: 0, true: 0, conf: 0.45993
[INFO    ] b'dog.7143.jpg'  : pred: 1, true: 1, conf: 0.89558
[INFO    ] b'cat.5792.jpg'  : pred: 1, true: 0, conf: 0.83258
[INFO    ] b'cat.8973.jpg'  : pred: 0, true: 0, conf: 0.23535
[INFO    ] b'dog.8704.jpg'  : pred: 0, true: 1, conf: 0.45196
```

#### train.log

> log regarding the training of the graph

```python
2018-07-05 18:54:19,199 - train_graph.py:100 -          train_graph()][INFO    ]: -> START epoch num: 1
[2018-07-05 18:54:19,235 - train_graph.py:110 -          train_graph()][DEBUG   ]: reset train and validation metric accumulators: [<tf.Operation 'metrics/val_metrics/val_met_reset_op' type=NoOp>, <tf.Operation 'metrics/val_loss_eval/val_loss_reset_op' type=NoOp>, <tf.Operation 'metrics/train_metrics/train_met_reset_op' type=NoOp>, <tf.Operation 'metrics/train_loss_eval/train_loss_reset_op' type=NoOp>]
[2018-07-05 18:54:19,248 - train_graph.py:118 -          train_graph()][DEBUG   ]: reinitialize training iterator: ./example/cats_v_dogs_01/data/record_holder/150/train.tfrecords
[2018-07-05 18:54:19,248 - train_graph.py:121 -          train_graph()][DEBUG   ]: -> START iterating training dataset
[2018-07-05 18:54:29,300 - train_graph.py:158 -          train_graph()][DEBUG   ]: [END] iterating training dataset
[2018-07-05 18:54:29,336 - train_graph.py:172 -          train_graph()][DEBUG   ]: reinitialize validation iterator: ./example/cats_v_dogs_01/data/record_holder/150/validation.tfrecords
[2018-07-05 18:54:29,337 - train_graph.py:174 -          train_graph()][DEBUG   ]: -> START iterating validation dataset
[2018-07-05 18:54:31,470 - train_graph.py:184 -          train_graph()][DEBUG   ]: [END] iterating validation dataset
[2018-07-05 18:54:31,481 - train_graph.py:190 -          train_graph()][INFO    ]: epoch 1 validation loss: 0.5908416509628296
[2018-07-05 18:54:31,568 - train_graph.py:195 -          train_graph()][DEBUG   ]: Model checkpoint saved in path: ./example/cats_v_dogs_01/trial_01/best_params/best_params_saver.ckpt
[2018-07-05 18:54:31,574 - train_graph.py:198 -          train_graph()][INFO    ]: best params saved: val acc: 68.175% val loss: 0.5908
[2018-07-05 18:54:31,601 - train_graph.py:218 -          train_graph()][INFO    ]: [END] epoch num: 1
[2018-07-05 18:54:31,601 - train_graph.py:100 -          train_graph()][INFO    ]: -> START epoch num: 2
```

## Getting Started

There is currently one example project [here](./examples/cats_v_dogs_01/README.md) that the main files implement.

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
- [sphinx](http://www.sphinx-doc.org/en/master/) documentation

### TODO: stretch

- An option for implementing ~smooth grad + for visualizations/ interpret-ability
- capsule layers (and scalar -> vect, vect -> scalar?)
- methodolgy for reducing model size for serving - plot performance as the dtype of operations are reduced / can some operations be removed from the graph?
- support k fold cross validation
- prepackaged models + params trained on well known datasets

#### Notes

Why YAML?

> Simply, because I wanted support for comments. Though x,y,or z may be ``better'' configuration types, yaml was selected for its general popularity and support for comments. I'm open to changing this in the future -- doing so would only require parsing the new config standard into the expect python dict -- but at the moment it isn't a top priority for me personally.  Additionally, since the interface/functionality will likely change dramatically, I think it would be best to stick with the one config file option (yaml) for now.
