import json
from pathlib import Path
from typing import List

import yeahml as yml


# a common set of layers was written to a pickle file - load these
# NOTE: the root of these common available names may belong in a fixture but I'm
# unsure how to do this at this point.
def return_data_and_ids(file_name: str) -> List[str]:
    ROOT_YML_DIR = Path(yml.__file__).parent
    CUR_AVAIL = (
        ROOT_YML_DIR.joinpath("config").joinpath("available").joinpath("components")
    )
    available_layers_path = CUR_AVAIL.joinpath(f"{file_name}.json")
    with open(available_layers_path, "r") as fp:
        data_dict = json.load(fp)
        available_layers = data_dict["data"]

    return available_layers
