# yamlflow

YamlFlow is a prototype (and yet to be fully implemented) framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).


## Getting Started

There is currently one example project [here](./example/cats_v_dogs_01/README.md) that the main files implement.

Main documentation + getting started will be created after a logical project structure is created and more examples have been tested (currently only one, cats vs dogs binary image classification).

Documentation for the main configuration file can be found [here](./documentation_helper/configuration_files/model_config.md). Documentation for the main architecture configuration file can be found [here](./documentation_helper/configuration_files/architecture.md) and documentation for the currently supported layer types [conv2d](./documentation_helper/configuration_files/layers/conv2d.md), [pooling](./documentation_helper/configuration_files/layers/pooling2d.md), and [dense](./documentation_helper/configuration_files/layers/dense.md) are also included.

 **If anyone would like to attempt to use or work on this, feel free to open an issue and/or reach out to me on twitter @Jack_Burdick**

## Motivation

The primary motivation is to define and create (simple) models easily. The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a seperate project (TODO: inlude link once made) that will attempt to generate/analyze architectures --- and use the yaml file as a way of generating/defining architectures.l

### Future Goals

At the moment the project is being developed around a binary image classification task. In the future, I'd like to support:

- Multi-class classification (Images)
- Regression
- Autoencoders
- GANs


### TODO and in Progress

- update tf_logs dir to be set on each run by a "name". include option to delete current logs as needed
- Add image augmentation logic
- add tensorboard scalar
- add graph to `tf_logs`
- include different metrics
- weight/bias regularization
- capsule layers (and scalar -> vect, vect -> scalar?)
- regression support (use Cali Housing)
- multiclass support (use MNIST)
- print model information prior to training
- add support for transfer learning (loading from a file, freezing)
- include documentation on what is possible (likely when the project is a little further along)