import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi


# NOTE: neither loss nor metric are required, and yet, without one it is useless
# the current config management does not allow for this.

PERFORMANCE = {
    "performance": {
        "objectives": {
            KPH("objective_name", multi=True): {
                KPH("loss", exact=True, required=False): {
                    "type": Text(),
                    KPH("options", exact=True, required=False): VPH("loss_options"),
                    KPH("track", exact=True, required=False): VPH("loss_track"),
                },
                KPH("metric", exact=True, required=False): {
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
