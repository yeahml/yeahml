from yaml_parse import create_model_and_hidden_config

# get relevant components from the yaml file/standardize
from yaml_parse import extract_dict_and_set_defaults

from build_graph import build_graph
from train_graph import train_graph
from eval_graph import eval_graph, eval_graph_from_saver
import os
import logging

# TODO: this needs to be handled differently
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_USE_CUDNN"] = "1" # necessary for Conv2d

## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?
# > maybe this belongs in "build graph?"

## Core Logic


## Config files for currently supported modes
# sigmoid_example = "./example/cats_v_dogs_01/model_config.yaml"
softmax_example = "./example/mnist/model_config.yaml"
model_config, hidden_config = create_model_and_hidden_config(softmax_example)
# model_config, hidden_config = extract_dict_and_set_defaults(model_config, hidden_config)

## build graph
g = build_graph(model_config, hidden_config)

## train graph
_ = train_graph(g, model_config, hidden_config)

## evaluate graph
# _ = eval_graph(g, model_config)

## same as above, but does not require manual graph creation
# > will load a graph from the saver path (if present)
_ = eval_graph_from_saver(model_config)


## Serving
## TODO: implementation
