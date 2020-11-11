import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi

DATA = {
    "data": {
        "datasets": {
            KPH("dataset_name"): {
                "in": {
                    KPH("feat_name", multi=True): {
                        "shape": Multi(),
                        "dtype": Text(),
                        KPH("startpoint", exact=True, required=False): Bool(),
                        KPH("endpoint", exact=True, required=False): Bool(),
                        KPH("label", exact=True, required=False): Bool(),
                    }
                },
                "split": {"names": Multi(element_types=Text())},
            }
        }
    }
}
