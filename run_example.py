import os
import logging

import yeahml as yml

# TODO: this needs to be handled differently
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_USE_CUDNN"] = "1" # necessary for Conv2d

## Core Logic

## Config files for currently supported modes
example = "./examples/cats_v_dogs/model_config.yaml"  # sigmoid example
# example = "./examples/mnist/model_config.yaml"  # softmax example
# example = "./examples/cali_housing/model_config.yaml"  # regression example
# example = "./examples/segmentation/model_config.yaml"  # binary segmentation example
###TODO: example = (
#     "./examples/multi_segmentation/model_config.yaml"
# )  # multi segmentation example
# example = "./examples/sentiment_imdb/model_config.yaml"  # sentiment analysis example
model_config, hidden_config = yml.create_model_and_hidden_config(example)


## build graph
g = yml.build_graph(model_config, hidden_config)

## train graph
_ = yml.train_graph(g, model_config, hidden_config)

## evaluate graph
# _ = eval_graph(g, model_config)

## same as above, but does not require manual graph creation
# > will load a graph from the saver path (if present)
yml.eval_graph_from_saver(model_config)


## Serving
## TODO: implementation
