import random


def select_optimizer(list_of_opt_names):

    # TODO: eventually, I'd like to pass loss/gradient information here and make
    # an adaptive decision

    if len(list_of_opt_names) == 1:
        selected_opt_name = list_of_opt_names[0]
    else:
        if len(list_of_opt_names) == 0:
            raise ValueError(f"trying to select optimizer from no optimizers")
        # naive approach
        ind = random.randint(0, len(list_of_opt_names) - 1)
        selected_opt_name = list_of_opt_names[ind]

    return selected_opt_name
