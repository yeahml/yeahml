import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi

HYPER_PARAMETERS = {
    "hyper_parameters": {
        "dataset": {
            "batch": Numeric(
                is_type=int, description="hyper_parameters:dataset:batch: <int>"
            ),
            KPH("shuffle_buffer", exact=True, required=False): Numeric(
                is_type=int,
                description="hyper_parameters:dataset:shuffle_buffer: <int>",
            ),
        },
        "epochs": Numeric(is_type=int, description="hyper_parameters:epochs: <int>"),
        KPH("early_stopping", exact=True, required=False): {
            "epochs": Numeric(is_type=int, description="patience"),
            "warm_up": Numeric(
                is_type=int,
                description="allow training to 'warm up' before keeping track",
            ),
        },
    }
}
