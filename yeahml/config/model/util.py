from typing import Any, Dict, List

from yeahml.config.model.config import IGNORE_HASH_KEYS


def make_hash(o: Dict[str, Any], ignore_keys: List[str] = []) -> int:
    # combination of several answers in
    # https://stackoverflow.com/questions/5884066/hashing-a-dictionary

    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """

    if isinstance(o, (set, tuple, list)):
        return tuple([make_hash(e) for e in o])
    elif not isinstance(o, dict):
        return hash(o)

    new_o = {}
    for k, v in o.items():
        if ignore_keys:
            if k not in ignore_keys:
                new_o[k] = make_hash(v, IGNORE_HASH_KEYS)
        else:
            new_o[k] = make_hash(v, IGNORE_HASH_KEYS)

    return hash(tuple(sorted(frozenset(new_o.items()))))
