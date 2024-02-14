import numpy as np

from .allocation import get_bundle_from_allocation_matrix
from .agent import BaseAgent
from .item import ScheduleItem

def utilitarian_welfare(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute utilitarian social welfare (USW), i.e., sum of utilities across all agents

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        float: USW / len(agents)
    """
    util = 0
    for agent_index in range(len(agents)):
        agent = agents[agent_index]
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        util += val
    return util / (len(agents))  


def nash_welfare(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute Nash social welfare (NSW), i.e., number of agents with 0 utility and product of utilities across all agents with utility>0

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        int: number of agents with utility 0
        float: n-root of NSW
    """
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


def leximin(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute Leximin vector, i.e. vector with agents utilities, sorted in decreasing order

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    valuations = []
    for agent_index in range(len(agents)):
        agent = agents[agent_index]
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        valuations.append(val)
    valuations.sort()
    valuations.reverse()
    return valuations


def EF_count(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    envy_count += 1
    return envy_count


def EF_agents(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    envy_count += 1
                    break
    return envy_count


def EF_1_count(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    there_is_no_item = True
                    for index, item in enumerate(other_bundle):
                        new_bundle = other_bundle.copy()
                        new_bundle.pop(index)
                        new_utility = agent.valuation(new_bundle)
                        if new_utility <= current_utility:
                            there_is_no_item = False
                            break
                    if there_is_no_item:
                        envy_count += 1
    return envy_count


def EF_1_agents(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    there_is_no_item = True
                    for index, item in enumerate(other_bundle):
                        new_bundle = other_bundle.copy()
                        new_bundle.pop(index)
                        new_utility = agent.valuation(new_bundle)
                        if new_utility <= current_utility:
                            there_is_no_item = False
                            break
                    if there_is_no_item:
                        envy_count += 1
                        break
    return envy_count


def EF_X_count(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    not_for_every_item = False
                    for index, item in enumerate(other_bundle):
                        new_bundle = other_bundle.copy()
                        new_bundle.pop(index)
                        new_utility = agent.valuation(new_bundle)
                        if current_utility < new_utility:
                            not_for_every_item = True
                            break
                    if not_for_every_item:
                        envy_count += 1
    return envy_count


def EF_X_agents(X, agents, items):
    envy_count = 0
    for agent_index in range(len(agents)):
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                agent = agents[agent_index]
                current_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_index
                )
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
                current_utility = agent.valuation(current_bundle)
                other_utility = agent.valuation(other_bundle)
                if current_utility < other_utility:
                    not_for_every_item = False
                    for index, item in enumerate(other_bundle):
                        new_bundle = other_bundle.copy()
                        new_bundle.pop(index)
                        new_utility = agent.valuation(new_bundle)
                        if current_utility < new_utility:
                            not_for_every_item = True
                            break
                    if not_for_every_item:
                        envy_count += 1
                        break
    return envy_count
