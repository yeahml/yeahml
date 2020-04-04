from yeahml.build.components.optimizer import return_available_optimizers
from yeahml.config.default.types.base_types import (
    categorical,
    list_of_categorical,
    numeric,
)
from yeahml.config.default.types.param_types import parameter_config
from typing import Dict, List, Tuple


class optimizer_config:
    def __init__(
        self, opt_type=None, opt_options=None, opt_name=None, opt_objectives=None
    ):

        # TODO: there are consistency issues here with the names of classes
        # and where the types are being created/checked

        self.type = categorical(
            default_value=None,
            required=True,
            is_in_list=return_available_optimizers(),
            to_lower=True,
        )(opt_type)

        self.options = parameter_config(
            known_dict={
                "learning_rate": numeric(
                    default_value=None, required=True, is_type=float
                )
            }
        )(opt_options)

        # TODO: in a secondary check, we need to ensure the losses specified
        # are valid+included
        self.objectives = list_of_categorical(
            default_value=None, required=True, to_lower=True
        )(opt_objectives)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self):
        return self.__dict__


class optimizers_parser:
    def __init__(self, optimizers_spec_dict=None):
        # TODO: this should be replaced by the __call__ logic
        if optimizers_spec_dict is None:
            self.conf_dict = None
        else:
            self.conf_dict = optimizers_spec_dict

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __call__(self, optimizers_spec_dict):
        # TODO: this should be moved to the __init__
        if isinstance(optimizers_spec_dict, dict):
            temp_dict = {}
            for k, d in optimizers_spec_dict.items():

                optimizer_name = categorical(
                    default_value=None, required=True, is_type=str, to_lower=False
                )(k)

                try:
                    opt_type = d["type"]
                except:
                    opt_type = None

                try:
                    opt_options = d["options"]
                except:
                    opt_options = None

                try:
                    opt_objectives = d["objectives"]
                except:
                    opt_objectives = None

                conf = optimizer_config(
                    opt_type=opt_type,
                    opt_options=opt_options,
                    opt_objectives=opt_objectives,
                )()
                temp_dict[k] = conf

        else:
            raise ValueError(
                f"{optimizers_spec_dict} is type {type(optimizers_spec_dict)} not type {type({})}"
            )
        return temp_dict


ACCEPTED_OPTERATORS = {"&": "joint", "|": "alternate", ",": "sequential"}


def create_instruction_step(instructions):
    # TODO: this is going to have to be seriously tested
    # maybe a state machine would be better? There is definitely a better way to
    # do this.

    steps = []
    left_indexes = []
    right_indexes = []
    arithmetic_indexes = []

    for i, o in enumerate(instructions):
        # open and close
        if o == "(":
            left_indexes.append(i)
        elif o == ")":
            right_indexes.append(i)

        # rest are assumed optimizer names or comments or instructions
        else:
            pass

    # build all steps of instructions
    for i, rv in enumerate(right_indexes):
        corresponding_l_paren_index = None

        # find corresponding index from left paren
        for j, cur_lv in enumerate(left_indexes):
            if cur_lv > rv:
                # we need the value before it becomes larger
                corresponding_l_paren_index = j - 1
        if not corresponding_l_paren_index:
            # will obtain furthest `(`
            corresponding_l_paren_index = len(left_indexes) - 1

        if corresponding_l_paren_index < 0:
            raise ValueError(
                f"no corresponding left index was found for the right ) at index {rv} - {instructions[:rv]}"
            )

        lv = left_indexes[corresponding_l_paren_index]
        left_indexes.remove(lv)
        steps.append((lv, rv))

    return steps


def create_instruction_order_dict(
    instructions: str, instruction_steps: List[Tuple[int, int]]
):
    """[summary]
    
    Parameters
    ----------
    instructions : str
        the instruction string
        e.g.
            "((a,b)&c)"
            where a, b, and c are the names of optimizers
    instruction_steps : Tuple[int, int]
        the indexes of parenthesis for a given instruction step
        e.g.
            [(1,5),(0,8)]
            which would correspond to "((a,b)&c)"
    
    Returns
    -------
    Dict[int: str]
        where the int is the instruction order and the string is the given instruction
    """

    instruct_dict = {}
    for i, step in enumerate(instruction_steps):
        cur_instruction_string = instructions[step[0] + 1 : step[1]]
        name_key = f"YEAHML_{i}"  # TODO: a hash would be more elegant?
        instruct_dict[name_key] = cur_instruction_string

    return instruct_dict


def parse_instruction_dict(instruction_dict):
    print(instruction_dict)
    print("*********" * 8)
    for key_name, instruct_str in instruction_dict.items():
        print(key_name)
        op_indexes = []
        for i, o in enumerate(instruct_str):
            if o in ACCEPTED_OPTERATORS.keys():
                op_indexes.append(i)
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
        optimizer_names = [o.lstrip("(").rstrip(")") for o in optimizer_names]

        print(optimizer_names)

    sys.exit()


def create_nested_instructs(instruction_dict: Dict[str, str]) -> Dict[str, str]:
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


def parse_instructions(instructions):
    # TODO: this is going to have to be seriously tested
    if not isinstance(instructions, str):
        raise TypeError(
            f"instructions ({instructions}) are of type {type(instructions)} not of type {type(str)}"
        )

    # sanity checks
    l_paren = instructions.count("(")
    r_paren = instructions.count(")")
    assert (
        l_paren == r_paren
    ), f"number of ( and ) parens not equal (={l_paren}, )={r_paren}"

    instruction_steps = create_instruction_step(instructions)

    # create instruction dict
    instruction_dict = create_instruction_order_dict(instructions, instruction_steps)

    # create nested instructions
    nested_instructs = create_nested_instructs(instruction_dict)
    print(nested_instructs)
    sys.exit()
    #####################################
    # HERE!!!
    #####################################

    # parse single instruction
    # parsed_instructions = parse_instruction_dict(instruction_dict)
    parsed_instructions = parse_instruction_dict(nested_instructs)

    print(parsed_instructions)
    sys.exit()

    return parsed_instructs


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

        instruct_dict = parse_instructions(instructions)

        return instruct_dict
