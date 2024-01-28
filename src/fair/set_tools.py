from itertools import chain, combinations


def powerset(iterable):
    """generates the power set of iterable

    Args:
        iterable (List or Set): ground set of items

    Returns:
        set of tuples: power set
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def is_mrf(ground, func):
    """checks 4 conditions of valid rank functions

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is an MRF, given ground; False otherwise
    """
    check1 = nonnegative_rank_value(ground, func)
    check2 = rank_value_leq_cardinality(ground, func)
    check3 = is_submodular(ground, func)
    check4 = is_monotonic_non_decreasing(ground, func)

    return check1 and check2 and check3 and check4


def is_submodular(ground, func):
    """performs check for submodularity of func, given a ground set of items

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is submodular, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        for j in range(len(powerset_ground)):
            add_val = func(powerset_ground[i]) + func(powerset_ground[j])
            int_val = func(
                list(set(powerset_ground[i]).intersection(set(powerset_ground[j])))
            )
            union_val = func(
                list(set(powerset_ground[i]).union(set(powerset_ground[j])))
            )
            if add_val < int_val + union_val:
                return False

    return True


def is_monotonic_non_decreasing(ground, func):
    """checks that func is monotonic non-decreasing, given the ground set, ground

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func is monotonic non_decreasing, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        for j in range(len(powerset_ground)):
            subset = False
            if set(powerset_ground[i]).issubset(set(powerset_ground[j])):
                subset = True
            if subset == True and func(powerset_ground[i]) > func(powerset_ground[j]):
                return False

    return True


def nonnegative_rank_value(ground, func):
    """checks that func always returns a non-negative value, given the ground set, ground

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function
    Returns:
        bool: True if func returns non-negative value, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        val = func(powerset_ground[i])
        if val < 0:
            return False
    return True


def rank_value_leq_cardinality(ground, func):
    """checks that func always returns a value less than the cardinality

    Args:
        ground (List): ground set of items (bundle)
        func (function): rank value function

    Returns:
        bool: True if func returns a value less than the cardinality, given ground; False otherwise
    """
    powerset_ground = list(powerset(ground))

    for i in range(len(powerset_ground)):
        val = func(powerset_ground[i])
        cardinality = len(powerset_ground[i])
        if val > cardinality:
            return False
    return True