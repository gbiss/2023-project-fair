import numpy as np

from .agent import BaseAgent
from .allocation import get_bundle_from_allocation_matrix
from .item import ScheduleItem


def precompute_bundles_valuations(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    bundles = [
        get_bundle_from_allocation_matrix(X, items, i) for i in range(len(agents))
    ]
    valuations = np.zeros((len(agents), len(agents)))
    for i, agent in enumerate(agents):
        for j, bundle in enumerate(bundles):
            valuations[i, j] = agent.valuation(bundle)
    return bundles, valuations


def EF_violations(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    valuations: type[np.ndarray] | None = None,
):
    """Compute envy-free violations.

    Compare every agent to all other agents, fill EF_matrix where EF_matrix[i,j]=1 if agent of index i
    envies agent of index j, 0 otherwise.

    Returns the number of EF violations, the number of envious agents, and EF_matrix

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        int: number of EF_violations
        int: number of envious agents
        EF_matrix (type[np.ndarray]): EF violations matrix as described above.
    """

    num_agents = len(agents)
    EF_matrix = np.zeros((num_agents, num_agents))

    if valuations is None:
        _, valuations = precompute_bundles_valuations(X, agents, items)

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            if valuations[i, i] < valuations[i, j]:
                EF_matrix[i, j] = 1
            if valuations[j, j] < valuations[j, i]:
                EF_matrix[j, i] = 1
    return np.sum(EF_matrix > 0), np.sum(np.any(EF_matrix > 0, axis=1))


def EF_1_count(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute EF-1 count

    Compare every agent to all other agents, add 1 to envy count if there is no item the second agent
    could drop that would make the first agent stop envying them.
    NOTE: we can add 1 multiple times for the same agents if they envy multiple other agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    envy_count = 0
    for agent_index, agent in enumerate(agents):
        current_bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        current_utility = agent.valuation(current_bundle)
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
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


def EF_1_agents(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute EF-1 agent count

    Compare every agent to all other agents, add 1 to envy count if there is no item a second agent
    could drop that would make the first agent stop envying them.
    NOTE: we can add 1 only once for every agent.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    envy_count = 0
    for agent_index, agent in enumerate(agents):
        current_bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        current_utility = agent.valuation(current_bundle)
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
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


def EF_X_count(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute EF-X count

    Compare every agent to all other agents, add 1 to envy count if there is at least one
    item that the second agents drops and the first agent still envies them.
    NOTE: we can add 1 multiple times for the same agents if they envy multiple other agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    envy_count = 0
    for agent_index, agent in enumerate(agents):
        current_bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        current_utility = agent.valuation(current_bundle)
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
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


def EF_X_agents(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute EF-X agent count

    Compare every agent to all other agents, add 1 to envy count if there is at least one
    item that another agent drops and the first agent still envies them.
    NOTE: we can add 1 only once for every agent.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        list[int]: utilities for all agents
    """
    envy_count = 0
    for agent_index, agent in enumerate(agents):
        current_bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        current_utility = agent.valuation(current_bundle)
        for agent_2_index in range(len(agents)):
            if agent_index != agent_2_index:
                other_bundle = get_bundle_from_allocation_matrix(
                    X, items, agent_2_index
                )
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
