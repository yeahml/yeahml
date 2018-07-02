# Model Configuration

The model configuration `.yaml` file is responsible for configuring high level training information. It is possible that this file will eventually be split into multiple configuration files.

TODO: Will be updated after sections have been more logical locations. As of right now a model spec may be similar to the following:


```yaml
# basic information
overall:
  name: 'cats vs dogs'
  # will print basic graph information to the terminal
  print_graph_spec: <bool>
  type: 'classification'
  options: 'binary' # currently only 'binary' is supported
  # for memory+time optimization/visualization in tensorboard
  # 'full' does both hardware and software
  trace: ['full','hardware','software'] # choose one

# information about the data being used
data:
  in_dim: [<int>, <int>, <int>] # [150, 150, 3]
  output_dim: [<int>] # [1], since only binary is supported at the moment
  # ./example/cats_v_dogs_01/data/record_holder/150
  TFR_dir: './path/to/tensorflow/flow/records'

# location to save the best parameters (graph weights)
saver:
  save_pparams: "<dir name : str>" # 'best_params'

# where should the logs be stored
tensorboard:
  log_dir: "<dir name : str>" # 'trial_01'

# model hyperparameters
hyper_parameters:
  lr: <int> # 0.00001
  batch_size: <int> # 16
  epochs: <int> # 20
  optimizer: ["adam","sgd","adadelta","adagrad","ftrl","rmsprop"] # choose one
  # activation function that will be used by default (if one isn't specified for the layer)
  default_activation: ["sigmoid","tanh","elu","selu","softplus","softsign","relu","relu6"] # choose one
  shuffle_buffer: <int> # 128

# What do the hidden layers look like?
hidden_config:
  yaml: './example/cats_v_dogs_01/hidden_config.yaml'

# information about the data for training + eval.. This heading is poorly chosen and will likely change
train:
  image_standardize: <bool> # True
  augmentation:
    aug_val: <bool> # True
    v_flip: <bool> # True
    h_flip: <bool> # True
```