from typing import Any, Dict, List, Tuple

ACCEPTED_OPTERATORS = {"&": "joint", "|": "alternate", ",": "sequential"}

"""
This entire script may be obsolete. It may make sense to explicitly write the
instruction/training "recipe" directly, with options specifying more advanced features.
"""


def _create_instrution_bounds(instructions: str) -> List[Tuple[int, int]]:
    """converts an instruction string into a list of bounds containing the 
    index of the opening and closing parenthesis
    
    Parameters
    ----------
    instructions : str
        e.g.
            '(((main_opt, second_opt) & second_opt & third_opt) & second_opt)'
    
    Returns
    -------
    List[Tuple[int, int]]
        e.g.
            [(2, 23), (1, 49), (0, 63)]
    
    Raises
    ------
    ValueError
        a string that does not have a closing ) for a (
    """

    steps = []
    left_indexes = []
    right_indexes = []

    for i, char in enumerate(instructions):
        # open and close
        if char == "(":
            left_indexes.append(i)
        elif char == ")":
            right_indexes.append(i)

        # rest are assumed optimizer names or comments or instructions
        else:
            pass

    # build all steps of instructions
    for i, right_ind in enumerate(right_indexes):
        corresponding_l_paren_index = None

        # find corresponding index from left paren
        for j, left_ind in enumerate(left_indexes):
            if left_ind > right_ind:
                # we need the value before it becomes larger
                corresponding_l_paren_index = j - 1
        if not corresponding_l_paren_index:
            # will obtain furthest `(`
            corresponding_l_paren_index = len(left_indexes) - 1

        if corresponding_l_paren_index < 0:
            raise ValueError(
                f"no corresponding left index was found for the right ) at index {right_ind} - {instructions[:right_ind]}"
            )

        lv = left_indexes[corresponding_l_paren_index]
        left_indexes.remove(lv)
        steps.append((lv, right_ind))

    return steps


def _create_instruction_order_dict(
    instructions: str, instruction_bounds: List[Tuple[int, int]]
):
    """parse and label each instruction step
    
    Parameters
    ----------
    instructions : str
        the instruction string
        e.g.
            '(((main_opt, second_opt) & second_opt & third_opt) & second_opt)'
    instruction_bounds : Tuple[int, int]
        the indexes of parenthesis for a given instruction step
        e.g.
            [(2, 23), (1, 49), (0, 63)]
    
    Returns
    -------
    Dict[int: str]
        where the int is the instruction order and the string is the given
        instruction
        e.g.
            {
                "YEAHML_0": "main_opt, second_opt",
                "YEAHML_1": "(main_opt, second_opt) & second_opt & third_opt",
                "YEAHML_2": "((main_opt, second_opt) & second_opt & third_opt) & second_opt",
            }

    """

    instruct_dict = {}
    for i, step in enumerate(instruction_bounds):
        cur_instruction_string = instructions[step[0] + 1 : step[1]]
        name_key = f"YEAHML_{i}"  # TODO: a hash would be more elegant?
        instruct_dict[name_key] = cur_instruction_string

    return instruct_dict


def _parse_instruction_dict(
    instruction_dict: Dict[str, str]
) -> Dict[str, Dict[str, List[str]]]:
    """Parse the individual string into the optimizers and operation.

    Note: only a single operation is allowed for a given string. Otherwise, a ()
    should have been used to group them first

    # TODO: testing
    
    Parameters
    ----------
    instruction_dict : Dict[str, str]
        [description]
        e.g.
            {
                "YEAHML_2": "(YEAHML_1)&second_opt",
                "YEAHML_1": "(YEAHML_0)&second_opt&third_opt",
                "YEAHML_0": "main_opt,second_opt",
            }
    
    Returns
    -------
    Dict[str, Dict[str : List[str]]]
        e.g.
            {
                "YEAHML_2": {"optimizers": ["YEAHML_1", "second_opt"], "operation": "&"},
                "YEAHML_1": {
                    "optimizers": ["YEAHML_0", "second_opt", "third_opt"],
                    "operation": "&",
                },
                "YEAHML_0": {"optimizers": ["main_opt", "second_opt"], "operation": ","},
            }
    
    Raises
    ------
    ValueError
        No operations were included in the instruction string
    ValueError
        Multiple operation types were included in the instruction string
    """

    parsed_instructions = {}
    for key_name, instruct_str in instruction_dict.items():
        parsed_instructions[key_name] = {}
        op_indexes = []
        op_types = []
        for i, char in enumerate(instruct_str):
            if char in ACCEPTED_OPTERATORS.keys():
                op_indexes.append(i)
                op_types.append(char)
            else:
                pass

        if not op_indexes:
            raise ValueError(f"no opperations were found in {instruct_str}")

        optimizer_names = []
        for i, op_i in enumerate(op_indexes):
            if i == 0:
                # set to -1 because we index by +1 in .append()
                prev_i = -1
            else:
                prev_i = op_indexes[i - 1]

            # obtain +1 beyond the prev instruction
            optimizer_names.append(instruct_str[prev_i + 1 : op_i])
        optimizer_names.append(instruct_str[op_i + 1 :])

        # strip `(` and `)`
        optimizer_names = [name.lstrip("( ").rstrip(" )") for name in optimizer_names]
        parsed_instructions[key_name]["optimizers"] = optimizer_names

        if not len(set(op_types)) == 1:
            raise ValueError(
                f"op types {set(op_types)} were discovered but only a single type is allowed"
            )
        parsed_instructions[key_name]["operation"] = op_types[0]

    return parsed_instructions


def _create_nested_instructs(instruction_dict: Dict[str, str]) -> Dict[str, str]:
    """ create a compact instruction set that nests the instructions. 
    Some instructions contain compound steps (i.e. contain entire steps that 
    should be done seperately). The visual example below explains this more
    clearly.
    
    Parameters
    ----------
    instruction_dict : Dict[str, str]
        dictionary containing the instruction name and instruction string
        e.g.
            {
                "YEAHML_0": "main_opt,second_opt",
                "YEAHML_1": "(main_opt,second_opt)&second_opt",
                "YEAHML_2": "((main_opt,second_opt)&second_opt)&second_opt",
            }
    
    Returns
    -------
    Dict[str, str]
        dictionary containing nested instructions
        e.g.
            {
                "YEAHML_2": "(YEAHML_1)&second_opt",
                "YEAHML_1": "(YEAHML_0)&second_opt",
                "YEAHML_0": "main_opt,second_opt",
            }

    """
    compact_dict = {}

    # loop the instructions keys backwards. they are numbered by the name and so
    # a reversed will work -- e.g. "YEAHML_1", "YEAHML_2", etc..
    # NOTE: the logic here is to remove the custom name str (YEAHML), split on
    # the split key ("_"), then obtain the str of a number and convert it to an
    # int for sorting
    key_list = list(instruction_dict.keys())
    first_k = key_list[0]
    split_key = "_"
    ind = len(first_k.split(split_key)[0]) + len(split_key)
    outter_to_inner = list(sorted(key_list, key=lambda x: int(x[ind:])))

    outter_to_inner.reverse()
    for i, k in enumerate(outter_to_inner):
        # skip final case
        if i + 1 < len(outter_to_inner):
            outer_str = instruction_dict[k]
            next_k = outter_to_inner[i + 1]
            inner_str = instruction_dict[next_k]

            # replace the inner string with the key
            # e.g.
            #   (main_opt,second_opt)&second_opt
            #   will become
            #   (YEAHML_N)&second_opt
            outer_str = outer_str.replace(inner_str, next_k)

            compact_dict[k] = outer_str
        else:
            # retain the final, un-nested instruction string
            compact_dict[k] = instruction_dict[k]

    return compact_dict


def parse_instructions(instructions: str) -> Dict[str, Dict[str, Any]]:
    """Parses the instruction string into a dictionary of operations with 
    associated optimizers
    
    Parameters
    ----------
    instructions : str
        e.g.
            "(((main_opt, second_opt) & second_opt & third_opt) & second_opt)"
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        dictionary containing the name of the instruction step and the
        associated optimizers and operation.
        e.g.
            {
                "YEAHML_2": {"optimizers": ["YEAHML_1", "second_opt"], "operation": "&"},
                "YEAHML_1": {
                    "optimizers": ["YEAHML_0", "second_opt", "third_opt"],
                    "operation": "&",
                },
                "YEAHML_0": {"optimizers": ["main_opt", "second_opt"], "operation": ","},
            }

    Raises
    ------
    TypeError
        Ensures instructions is of type string
    """
    if not isinstance(instructions, str):
        raise TypeError(
            f"instructions ({instructions}) are of type {type(instructions)} not of type {type(str)}"
        )

    # sanity checks
    l_paren = instructions.count("(")
    r_paren = instructions.count(")")
    assert (
        l_paren == r_paren
    ), f"number of `(` and `)` parens not equal `(`: {l_paren}, `)`: {r_paren}"

    # chain of functions to parse individual operations
    instruction_bounds = _create_instrution_bounds(instructions)
    instruction_dict = _create_instruction_order_dict(instructions, instruction_bounds)
    nested_instructs = _create_nested_instructs(instruction_dict)
    parsed_instructions = _parse_instruction_dict(nested_instructs)

    return parsed_instructions


class instruct_parser:
    def __init__(self, instruct_spec_dict=None):
        # NOTE: currently unused
        if instruct_spec_dict is None:
            self.instruct_spec_dict = None
        else:
            self.instruct_spec_dict = optimizers_spec_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, optimizers_spec_dict):
        # TODO: this should be moved to the __init__

        try:
            instructions = optimizers_spec_dict["instructions"]
        except KeyError:
            raise ValueError(
                f"no instructions were found in {optimizers_spec_dict.keys()}"
            )

        if not instructions:
            raise ValueError(
                f"No instructions were found in optimizers_spec_dict: {optimizers_spec_dict['instructions']}"
            )

        instruct_dict = parse_instructions(instructions)

        return instruct_dict
