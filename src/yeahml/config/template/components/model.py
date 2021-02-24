import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi
from crummycm.validation.types.values.compound.either import Either

MODEL = {
    "model": {
        "name": Text(),
        "start_fresh": Bool(),
        "layers": {
            KPH("layer_name", multi=True): {
                "type": Text(),
                KPH("source", required=False, exact=True): Text(),
                KPH("options", required=False, exact=True): {
                    KPH("layer_option_key", multi=True): VPH("layer_option_value"),
                    KPH("activation", exact=True, required=False): {
                        "type": Text(),
                        KPH("options_key", required=False, multi=True): VPH(
                            "options_value"
                        ),
                    },
                },
                # TODO: this could be Either a list of strings or string
                KPH("in_name", exact=True, required=False): Either(
                    either_seq=[Text(required=False), Multi(required=False)],
                    return_as_type=list,
                ),
                KPH("endpoint", exact=True, required=False, populate=True): Bool(
                    default_value=False
                ),
                KPH("startpoint", exact=True, required=False, populate=True): Bool(
                    default_value=False
                ),
            }
        },
    }
}
