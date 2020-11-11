import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi

CALLBACKS = {
    KPH("callbacks", exact=True, required=False): {
        "objects": {
            KPH("callback_name", multi=True): {
                "type": Text(to_lower=True),
                KPH("options", exact=True, required=False): {
                    KPH("options_key", multi=True, required=False): VPH("options_value")
                },
            }
        }
    }
}
