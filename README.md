# YeahML

YeahML is a prototype library for building, training, and analyzing neural
networks using primarily configuration files (raw python code is also accepted)

Please note: 
- This is a work in progress and may change dramatically
- Feedback is welcome (please open an issue)
- The current implementation is slow (e.g. roughly ~40+% slower on a supervised
  mnist classification example)


## [Examples](./examples)

[Examples](./examples) are a work in progress, but do show the basic functionality.


## Motivation

The primary motivation is to define and create models easily. The
*real*/secondary purpose for this framework is to act as a foundation for a
separate project that will attempt to generate/analyze architectures by allowing
for config based model and training definitions. Additionally, the training loop
is designed to accommodate multiple datasets and aims to ease
multi/meta-learning efforts (but is in early stages).