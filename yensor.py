from yaml_parse import create_model_and_arch_config
from yaml_parse import extract_from_dict
from build_graph import build_graph
from train_graph import train_graph

# parse config files
MC, AC = create_model_and_arch_config("./experiment/cats_v_dogs_01/model_config.yaml")
MCd, ACd = extract_from_dict(MC, AC)
# print(MCd)
# print(ACd)

# TODO: There should be some ~basic error checking here against design
# do the metrics make sense for the problem? layer order? est. param size?

# build graph (incomplete)
# TODO: I'd like only ACd to be passed here
g = build_graph(MCd, ACd)
print("done construction")
# print(g)

# train graph (incomplete)
# TODO: I'd like to return err here. but I'm not sure this is the best way to handle it
_ = train_graph(g, MCd)
print("done training")

# evaluate graph (incomplete)


# Serving?
