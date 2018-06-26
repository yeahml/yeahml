from yaml_parse import create_model_and_arch_config
from yaml_parse import extract_from_dict
from build_graph import build_graph
from train_graph import train_graph
from eval_graph import eval_graph
import os

## parse config files
MC, AC = create_model_and_arch_config("./experiment/cats_v_dogs_01/model_config.yaml")
MCd, ACd = extract_from_dict(MC, AC)


# TODO: this needs to be handled differently
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_USE_CUDNN"] = "1" # necessary for Conv2d


## Basic Error Checking
# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?


## build graph
# TODO: I'd like only ACd to be passed here
g = build_graph(MCd, ACd)


## train graph
# TODO: I'd like to return err here. but I'm not sure this is the best way to handle it
_ = train_graph(g, MCd)


## evaluate graph (incomplete)
# TODO: only creating a new graph to ensure no issues
g_e = build_graph(MCd, ACd)
_ = eval_graph(g_e, MCd)


## Serving
## TODO: implementation
