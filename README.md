# yamlflow

YamlFlow is a prototype (and yet to be fully implemented) framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml).

The primary motivation is to define and create (simple) models easily. The *real* purpose for this framework, in addition to making developing/training models more easily, is to act as a helper for a seperate project (TODO: inlude link once made) that will attempt to generate/analyze architectures --- and use the yaml file as a way of generating/defining architectures.

## Getting Started

There is currently one example project (TODO: link) -- after a project structure has been created, documentation and improved documentation will be included. The current example expects the tf records to already be included on the local environment (these can be created by downloading the kaggle datset and following the included `.py` files). After the tf records have been created. The current directory and config files will work as expected when calling `python yamlflow.py`. **If anyone would like to attempt to use or work on this, feel free to open an issue and/or reach out to me on twitter @Jack_Burdick**

### TODO

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