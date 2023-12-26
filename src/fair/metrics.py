import numpy as np

from .allocation import get_bundle_from_allocation_matrix


def utilitarian_welfare(X, agents, items):
    util = 0
    for agent_index in range(len(agents)):
        agent = agents[agent_index]
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        util += val
    return util / (
        len(agents)
    )  # The number of agents is given by dim(X[(0)])-1, so as to not consider agent 0


def nash_welfare(X, agents, items):
    util = 0
    num_zeros = 0
    for agent_index in range(len(agents)):
        agent = agents[agent_index]
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        if val == 0:
            num_zeros += 1
        else:
            util += np.log(val)
    return num_zeros, np.exp(util / (len(agents) - num_zeros))


def leximin(X, agents, items):
    valuations = []
    for agent_index in range(len(agents)):
        agent = agents[agent_index]
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        valuations.append(val)
    valuations.sort()
    valuations.reverse()
    return valuations
