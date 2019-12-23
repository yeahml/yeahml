import os

import yeahml as yml
from yeahml.log.yf_logging import config_logger  # custom logging

# TODO: this needs to be handled differently
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_USE_CUDNN"] = "1" # necessary for Conv2d

## Core Logic

## Config files for currently supported modes
# NOTE the model_cdict.yaml will need to be updated to model_cdict.yaml
# I'm leaving the names as is as a way to mark which examples I need to update
# example = "./examples/cats_v_dogs/model_cdict.yaml"  # sigmoid example
example = "./examples/mnist/main_config.yaml"  # softmax example
# example = "./examples/cali_housing/model_cdict.yaml"  # regression example
# example = "./examples/segmentation/model_cdict.yaml"  # binary segmentation example
# example = (
#     "./examples/multi_segmentation/model_cdict.yaml"
# )  # multi segmentation example
# example = "./examples/sentiment_imdb/model_cdict.yaml"  # sentiment analysis example
config_dict = yml.create_configs(example)

## build graph
model = yml.build_model(config_dict)
print(model.summary())

## train graph
train_dict = yml.train_model(model, config_dict)
print(train_dict)

# ## evaluate graph
# # yml.eval_graph(g, model_cdict) # not currently implemented (use eval_graph_from_saver)
# # same as eval_graph(), but will not require manual graph creation
# # > will load a graph from the saver path (if present)
# "./examples/mnist/saved_params/best_params_saver.h5"
eval_dict = yml.eval_model(model, config_dict)
print(eval_dict)
