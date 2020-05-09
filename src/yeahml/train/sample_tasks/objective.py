import random


def select_objective(loss_objective_names):

    # TODO: eventually, I'd like to pass loss/gradient information here and make
    # an adaptive decision

    if len(loss_objective_names) == 1:
        select = loss_objective_names[0]
    else:
        # naive approach
        ind = random.randint(0, len(loss_objective_names) - 1)
        select = loss_objective_names[ind]

    return select
