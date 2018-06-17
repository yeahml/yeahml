from yaml_parse import create_model_and_arch_config
from yaml_parse import extract_from_dict
from build_graph import build_graph

# parse config files
MC, AC = create_model_and_arch_config("./experiment/cats_v_dogs_01/model_config.yaml")
MCd, ACd = extract_from_dict(MC, AC)
# print(MCd)
# print(ACd)

# build graph (incomplete)
g = build_graph(MCd, ACd)
# print(g)

# train graph (incomplete)
#

# evaluate graph (incomplete)


# Serving?
