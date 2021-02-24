import crummycm as ccm
from yeahml.config.template.components.meta import META
from yeahml.config.template.components.performance import PERFORMANCE
from yeahml.config.template.components.hyper_parameters import HYPER_PARAMETERS
from yeahml.config.template.components.logging import LOGGING
from yeahml.config.template.components.data import DATA
from yeahml.config.template.components.model import MODEL
from yeahml.config.template.components.optimize import OPTIMIZE
from yeahml.config.template.components.callbacks import CALLBACKS


"""
TODO: 
- rethink naming, headings, etc....
- descriptions
- not sure what to do about `epochs`

think on
--------
preprocess
augment

"""


TEMPLATE = {}
TEMPLATE = {**TEMPLATE, **META}
TEMPLATE = {**TEMPLATE, **PERFORMANCE}
TEMPLATE = {**TEMPLATE, **HYPER_PARAMETERS}
TEMPLATE = {**TEMPLATE, **LOGGING}
TEMPLATE = {**TEMPLATE, **DATA}
TEMPLATE = {**TEMPLATE, **MODEL}
TEMPLATE = {**TEMPLATE, **OPTIMIZE}
# optional
TEMPLATE = {**TEMPLATE, **CALLBACKS}

# p = "/home/jackburdick/dev/github/YeahML/src/yeahml/config/template/template.yml"
# _ = ccm.template(TEMPLATE, p)

