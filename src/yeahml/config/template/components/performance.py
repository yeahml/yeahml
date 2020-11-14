import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi

PERFORMANCE = {
    "performance": {
        "objectives": {
            KPH("objective_name", multi=True): {
                "loss": {
                    "type": Text(),
                    KPH("options", exact=True, required=False): VPH("loss_options"),
                    KPH("track", exact=True, required=False): VPH("loss_track"),
                },
                "metric": {
                    "type": Text(),
                    KPH("options", exact=True, required=False): VPH("loss_options"),
                    KPH("track", exact=True, required=False): VPH("loss_track"),
                },
                "in_config": {
                    "type": Text(),
                    KPH("options", exact=True, required=False): {
                        "prediction": Text(),
                        "target": Text(),
                    },
                    "dataset": Text(),
                },
            }
        }
    }
}
