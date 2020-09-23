from pkg_resources import get_distribution, DistributionNotFound
import os.path


# this block is adapted from
# https://stackoverflow.com/questions/17583443/what-is-the-correct-way-to-share-package-version-with-setup-py-and-the-package
# I'm not 100% sure this is the best way to do this yet
try:
    _dist = get_distribution("yeahml")
    print("HEY KEVIN", _dist)
    # Normalize case for Windows systems
    dist_loc = os.path.normcase(_dist.location)
    here = os.path.normcase(__file__)
    print("HEY KEVIN", here, dist_loc)
    if not here.startswith(os.path.join(dist_loc, "src/yeahml")):
        # not installed, but there is another version that *is*
        raise DistributionNotFound
except DistributionNotFound:
    __version__ = "Please install this project with setup.py (e.g.`pip install ./YeahML/` or `pip install -e ./YeahML/`)"
else:
    __version__ = _dist.version


# Config
from yeahml.config.create_configs import create_configs

# build model
from yeahml.build.build_model import build_model

# train
from yeahml.train.train_model import train_model

# evaluate
from yeahml.evaluate.eval_model import eval_model

# visualize training
from yeahml.visualize.tracker import basic_plot_tracker
