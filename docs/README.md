# Docs

TODO:

## Example in code

```python
import yeahml as yml

######################################################
# create+validate configuration file
yml_config = yml.create_configs("./main_config.yml")

######################################################
# build a model according to the config
model = yml.build_model(yml_config)


######################################################
# train the created model, where `train_dict` contains 
# information about the training dynamics
train_dict = yml.train_model(model, yml_config)

# note: if you can also pass an already created tf_dataset
# e.g.
# train_dict = yml.train_model(model, yml_config, ds_dict)
# where ds_dict contains may look like:
# ds_dict = {
#   "ds_name":{
#     "train": <tf.dataset>, 
#     "val": <tf.dataset>, 
#     "test": <tf.dataset>
#     }
#   }

######################################################
# evaluate graph -- can specify parameters to use, or will load the 
# "best params" (as defined by the user) created during training.
# where `eval_dict` contains information about the performance on the test dataset
eval_dict = yml.eval_model(
    model,
    yml_dict,
    dataset=ds_dict["test"]
)

```



## Configuration Files

The configuration files can either be written in json or yaml (hence the project
name..) and may look something like the following. Eventually, the documentation
will display all of the available options.

<!-- Where documentation+examples for the main configuration file can be found [here](./docs/configuration_files/model_cdict.md) and documentation+examples for the main hidden layer architecture configuration file can be found [here](./docs/configuration_files/hidden_config.md). -->

The main config may look similar to the following. Where each of the main
headings can either be defined in the main configuration file or may be specified
as a path to another configuration file (such as the example case below where
the model is defined in another location). These indicated heading are required:
- meta
  - defines organizational information (project name etc)
- logging
  - defines how and where to log tf/yeahml information
- performance
  - defines the metrics/loss information for training/evaluating the model
- data
  - defines the connection from raw data --> model (may change significantly)
- optimize
  - defines _how_ to train the model
- hyper_parameters
  - catch-all for adjusting training information like batch_size.. however,
    these will should be moved to their corresponding locations. for example,
    batch size likely belongs with the dataset.
- model
  - defines how the model should be built and connected.  All tf.keras layers
    are available by their name.

```yaml
meta:
  data_name: "mnist"
  experiment_name: "trial_00"
  start_fresh: False

logging:
  console:
    level: "info"
    format_str: null
  file:
    level: "ERROR"
    format_str: null
  track:
    tracker_steps: 30
    tensorboard:
      param_steps: 50
  graph_spec: True

performance:
  objectives:
    main_obj:
      loss:
        type: "sparse_categorical_crossentropy"
        track: "mean"
      metric:
        type: ["SparseCategoricalAccuracy"]
        options: [null]
      in_config:
        type: "supervised"
        options:
          prediction: "y_pred"
          target: "y_target"
        dataset: "mnist"

data:
  datasets:
    "mnist":
      in:
        x_image:
          shape: [28, 28, 1]
          dtype: "float32" # this is a cast
        y_target:
          shape: [1, 1]
          dtype: "int32"
          label: True
      split:
        names: ["train", "val"]

optimize:
  optimizers:
    "main_opt":
      type: "adam"
      options:
        learning_rate: 0.0001
      objectives: ["main_obj"]
  directive:
    instructions: "main_opt"

hyper_parameters:
  epochs: 5
  dataset:
    batch: 16
    shuffle_buffer: 128

model:
  path: "./model_config.yml"
```

A basic model config (where the path to this file is specified above by (`model:path`) may look similar to the following:

```yaml
name: "model_a"
start_fresh: True

layers:
  conv_1:
    type: "conv2d"
    options:
      filters: 8
      kernel_size: 3
      padding: "same"
    in_name: "x_image"
  conv_2_downsample:
    type: "conv2d"
    options:
      filters: 8
      kernel_size: 3
      strides: 2
      padding: "same"
  conv_3:
    type: "conv2d"
    options:
      filters: 8
      kernel_size: 3
      strides: 1
      padding: "same"
  conv_4_downsample:
    type: "conv2d"
    options:
      filters: 8
      kernel_size: 3
      strides: 2
      padding: "same"
  flatten_1:
    type: "flatten"
  dense_1:
    type: "dense"
    options:
      units: 128
      activation:
        type: "elu"
  dense_2:
    type: "dense"
    options:
      units: 32
      activation:
        type: "elu"
  y_pred:
    type: "dense"
    options:
      units: 10
      activation:
        type: "softmax"
    endpoint: True
```

### Sightly more advanced Example
The segmentation [example](./examples/segmentation_oxford_pets) shows how to use
custom written layers. Please note, it is also possible to mix and match
predefined layers as well as the custom layers

```yaml
layers:
  #....
  type: "down_block"
    source: "layer/block_module.py"
    options:
      filters: 32
      down_size: 2 # out: 32x32
      activation:
        type: elu

```

where a python script (indicated by the `source:`) contains a custom layer named
"down_block" (indicated by the `type:`). which can contain more custom
subclassed layers (shown by n_by_n, etc.)

```python
import tensorflow as tf

from .base_components import n_by_n
from .block_components import multipath, multipath_reduction

class down_block(tf.keras.layers.Layer):
    def __init__(
        self, filters=None, down_size=None, padding="same", activation=None, **kwargs
    ):
        if not filters:
            raise ValueError("filters are required")
        if not down_size:
            raise ValueError("down_size is required")
        if not padding:
            raise ValueError("padding is required")
        if not activation:
            raise ValueError("activation is required")
        self.down_size = down_size
        super(down_block, self).__init__(**kwargs)
        self.conv_a = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )
        self.conv_b = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )
        self.concat = tf.keras.layers.Concatenate()
        self.conv_1x1 = n_by_n(
            kernel_size=1,
            filters=filters,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.down = multipath_reduction(
            filters=filters,
            strides=down_size,
            padding=padding,
            activation=activation,
            **kwargs
        )
        self.conv_out = multipath(
            filters=filters, padding=padding, activation=activation, **kwargs
        )

    def get_config(self):
        config = super(up_block, self).get_config()
        config.update({"down_size": self.down_size})
        return config

    def call(self, inputs):

        out_a = self.conv_a(inputs)
        out_a = self.conv_b(out_a)
        out_b = self.concat([out_a, inputs])
        out = self.conv_1x1(out_b)

        out = self.down(out)
        out = self.conv_out(out)

        return out
```



### TensorBoard

After training, [tensorboard](https://www.tensorflow.org/tensorboard) can be
used to inspect the graph and metrics by issuing the following command from the
appropriate subdirectory:
`tensorboard --logdir "tf_logs/"` which will open tensorboard and display
figures similar to those below.




<!-- ### Logging

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
  - Contains information about training the graph [Information + Example](./docs/logs/train.md) -->