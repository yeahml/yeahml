from crummycm.validation.types.placeholders.placeholder import (
    KeyPlaceholder as KPH,
    ValuePlaceholder as VPH,
)
from crummycm.validation.types.values.element.numeric import Numeric
from crummycm.validation.types.values.element.text import Text
from crummycm.validation.types.values.element.bool import Bool
from crummycm.validation.types.values.compound.multi import Multi


META = {
    "meta": {
        KPH("yeahml_dir", exact=True, required=False, populate=True): Text(
            default_value="yeahml",
            description=(
                "Root directory to store information\n"
                " > e.g. meta:yeahml_dir: 'yeahml'"
            ),
        ),
        "data_name": Text(
            description=(
                "Description of the data used \n"
                " > e.g. meta:data_name: 'mnist', or  meta:data_name: 'V00'\n"
                "this logic will likely change in the future"
            )
        ),
        "experiment_name": Text(
            description=(
                "Name for the experiment being performed\n"
                " > e.g. meta:experiment_name: 'trial_00'"
            )
        ),
        "start_fresh": Bool(
            default_value=False,
            description=(
                "Used to determine whether previous experiments should be deleted\n"
                " > e.g. meta:start_fresh: True"
            ),
        ),
        KPH("random_seed", exact=True, required=False): Numeric(
            is_type=int,
            description=(
                "Used to set the random seed for tensorflow\n"
                " > e.g. meta:rand_seed: 42"
            ),
        ),
        KPH("default_load_params_path", exact=True, required=False): Text(),
        # TODO: tracing
        # "trace_level": Text(description="meta:trace_level: <str>"),
    }
}
