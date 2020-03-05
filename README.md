# YeahML

YeahML is a prototype framework for creating ML models ([D|R|C]NNs, primarily) using only, easy to understand, with sane defaults, configuration files (yaml or json).

Examples are currently being worked through and can be found in the [examples directory](./examples)

The core implementation is as follows:

1. Write configuration file(s) (json or yaml, see below)
2. Use python, as shown below, to train and evaluate the model
3. Iterate
  - logs/tensorboard are created


## Main use

```python
example = "./main_config.yml"
yml_dict = yml.create_configs(example)

# build graph
model = yml.build_model(yml_dict)

# train graph
ds_tuple = (ds_dict["train"], ds_dict["val"])
train_dict = yml.train_model(model, yml_dict, ds_tuple)

# evaluate graph -- will load the "best params"
eval_dict = yml.eval_model(
    model,
    yml_dict,
    dataset=ds_dict["test"]
)

```

<!-- Where documentation+examples for the main configuration file can be found [here](./docs/configuration_files/model_cdict.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./docs/configuration_files/hidden_config.md). -->

## [Examples](./examples)

Examples are a work in progress


### Configuration Files

The model config may look similar to the following:

```yaml
meta:
  data_name: 'abalone'
  experiment_name: 'trial_00'

logging:
  console:
    level: 'info'
    format_str: null
  file:
    level: 'ERROR'
    format_str: null
  graph_spec: True

performance:
  objectives:
    main:
      loss: 
        type: 'MSE'
      metric:
        type: ["MeanSquaredError", "MeanAbsoluteError"]
        options: [null, 
                  null]
      in_config:
        type: "supervised"
        options:
          prediction: "dense_out"
          target: "target_v"

data:
  in:
    features:
      shape: [2,1]
      dtype: 'float64'
    target_v:
      shape: [1]
      dtype: 'int32'

hyper_parameters:
  optimizer: 
    type: 'adam'
    options:
      learning_rate: 0.0001
      beta_1: 0.91
  epochs: 30
  dataset:
    # TODO: I would like to make this logic more abstract
    # I think the only options that should be applied here are "batch" and "shuffle"
    batch: 16
    shuffle_buffer: 128 # this should be grouped with batchsize
model:
  path: './model_config.yml'
```

A basic model config (where the path to this file is specified above by (`model:path`) may look similar to the following:

```yaml
meta:
  name: "model_a"
  name_override: True
  activation:
    type: 'elu'

layers:
  dense_1:
    type: 'dense'
    options:
      units: 16
    in_name: 'features'
  dense_2:
    type: 'dense'
    options:
      units: 8
  dense_out:
    type: 'dense'
    options:
      units: 1
      activation:
        type: 'linear'
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