import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi
from yeahml.build.components.optimizer import return_available_optimizers

OPTIMIZE = {
    "optimize": {
        "optimizers": {
            KPH("optimizer_name", multi=True): {
                "type": Text(
                    is_in_list=return_available_optimizers(),
                    to_lower=True,
                    description=(
                        "The type of optimizer being used\n"
                        " > e.g. optimize:optimizers:'name':type: 'adam'"
                    ),
                ),
                KPH("options", exact=True, required=False): {
                    "learning_rate": Numeric(is_type=float),
                    KPH("other_options", required=False): VPH("optimizer_options"),
                },
                "objectives": VPH("optimizer_objectives"),
            }
        }
        # "directive": {"instructions": "SS"},
    }
}
