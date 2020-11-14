import crummycm as ccm
from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi

from yeahml.build.components.dtype import return_available_dtypes

DATA = {
    "data": {
        "datasets": {
            KPH("dataset_name", multi=True): {
                "in": {
                    KPH("feat_name", multi=True): {
                        "shape": Multi(element_types=int),
                        "dtype": Text(is_in_list=return_available_dtypes()),
                        KPH(
                            "startpoint", exact=True, required=False, populate=True
                        ): Bool(default_value=True),
                        KPH(
                            "endpoint", exact=True, required=False, populate=True
                        ): Bool(default_value=False),
                        KPH("label", exact=True, required=False, populate=True): Bool(
                            default_value=False
                        ),
                    }
                },
                "split": {"names": Multi(element_types=Text())},
            }
        }
    }
}
