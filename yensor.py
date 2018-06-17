from yaml_parse import create_model_and_arch_config
from build_graph import build_graph

# parse config files
MC, AC = create_model_and_arch_config("./model_config.yaml")
# print(MC)
# print(AC)

# build graph (incomplete)
custom_graph = build_graph(MC, AC)

# train graph (incomplete)


# evaluate graph (incomplete)


# Serving?
