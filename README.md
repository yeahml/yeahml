# YeahML

YeahML is a prototype library for building, training, and analyzing neural networks using configuration files

Please note: 
- This is a personal project and is a work in progress and may change
  dramatically
- I would appreciate help developing+thinking through this project. If you are
  interested, please reach out via an issue or PR
- The current implementation is unbearably slow (roughly ~40+% slower on a
  supervised mnist classification example), but presently I'm focused on
  cleaning up the base to be more readable/correct.


## [Examples](./examples)

[Examples](./examples) are a work in progress, but show the basic functionality.


## Motivation

The primary motivation is to define and create models easily. The *real* purpose
for this framework, in addition to developing+training models more easily, is to
act as a foundation for a separate project that will attempt to generate/analyze
architectures (AutoML) -- by allowing for config based model and training
definitions. Additionally, the training loop is designed to accommodate multiple
datasets and aims to ease multi/meta-learning efforts (but is in early stages).