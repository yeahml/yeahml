# YamlFlow

[//]: # (Image References)
[tensorboard_graph]: ./misc/tensorboard_graph.png
[tensorboard_scalar]: ./misc/tensorboard_scalar.png

YamlFlow is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).

The goal of the core implementation is as follows:

Where documentation+examples for the main configuration file can be found [here](./documentation_helper/configuration_files/model_config.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./documentation_helper/configuration_files/hidden_config.md). Additional information, such as documentation for the currently supported layer types [conv2d](./documentation_helper/configuration_files/layers/conv2d.md), [pooling](./documentation_helper/configuration_files/layers/pooling2d.md), and [dense](./documentation_helper/configuration_files/layers/dense.md) are also included.

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

The model config may look similar to the following:

```yaml
overall:
  name: 'cats vs dogs'
  print_graph_spec: True
  type: 'classification'
  options: 'binary'
  trace: 'full'
data:
  in_dim: [150, 150, 3]
  output_dim: [1]
  TFR_dir: './example/cats_v_dogs_01/data/record_holder/150'
saver:
  save_pparams: 'best_params'
tensorboard:
  log_dir: 'trial_01'
hyper_parameters:
  lr: 0.00001
  batch_size: 16
  epochs: 20
hidden:
  yaml: './example/cats_v_dogs_01/hidden_config.yaml'
implementation:
  optimizer: 'adam'
  default_activation: 'elu'
  shuffle_buffer: 128
train:
  image_standardize: True
  augmentation:
    aug_val: True
    v_flip: True
    h_flip: True
```

The hidden layer architecture config (where the path to this file is specified above by (`hidden:yaml`) may look similiar to the following:

```yaml
layers:
  conv_1:
    type: 'conv2d'
    options:
      filters: 16
      kernel_size: 3
      strides: 2
  pool_1:
    type: 'pooling2d'
    options:
      pool_type: "avg"
  conv_2:
    type: 'conv2d'
    options:
      filters: 32
      kernel_size: 3
      strides: 1
  pool_2:
    type: 'pooling2d'
  conv_3:
    type: 'conv2d'
    options:
      filters: 64
      kernel_size: 3
      strides: 1
  pool_3:
    type: 'pooling2d'
    options:
      pool_type: "max"
      dropout: 0.5
  dense_1:
    type: 'dense'
    options:
      units: 64
      dropout: 0.5
  dense_2:
    type: 'dense'
    options:
      units: 16
      dropout: 0.5
```

After training, tensorboard can be used to inspect the graph and metrics by issuing the following command: `tensorboard --logdir "tf_logs/"` which will open tensorboard and display figures similar to those below.

![Example of TensorFlow graph in tensorboard -- showing the graph and compute time trace][tensorboard_graph]
Where the image on the left is of the specified graph (in the architecture config yaml file) and the graph on the right is showing a trace of compute time of the operations during training (set by the `overall:trace`) parameter in the model config file.

![Example of TensorFlow metrics in tensorboard -- showing scalars and histogram][tensorboard_scalar]

Where the image on the right shows the training and validation metrics during training (computed over the entire iteration) and the right shows histograms of the parameters (weights+biases) calculated during training.

## Getting Started

There is currently one example project [here](./example/cats_v_dogs_01/README.md) that the main files implement.

Main documentation + getting started will be created after a logical project structure is created and more examples have been tested (currently only one, cats vs dogs binary image classification). **If anyone would like to attempt to use or work on this, feel free to open an issue and/or reach out to me on twitter @Jack_Burdick**

## Motivation

The primary motivation is to define and create (simple) models easily (for machines and humans). The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a seperate project (TODO: inlude link once made) that will attempt to generate/analyze architectures.

### Future Goals

At the moment the project is being developed around a binary image classification task. In the future, I'd like to support:

- Multi-class classification (Images)
- Regression
- Autoencoders
- GANs

### TODO and in Progress

- update tf_logs dir to be set on each run by a "name". include option to delete current logs as needed
- add graph to `tf_logs`
- include different metrics
- weight/bias regularization
- capsule layers (and scalar -> vect, vect -> scalar?)
- regression support (use Cali Housing)
- multiclass support (use MNIST)
- add support for transfer learning (loading from a file, freezing)
- include documentation on what is possible (likely when the project is a little further along)