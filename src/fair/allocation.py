import time
from queue import Queue

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .agent import BaseAgent
from .item import ScheduleItem

"""Initializations functions"""


def initialize_allocation_matrix(items: list[ScheduleItem], agents: list[BaseAgent]):
    """Initialize allocation matrix.
    Initially, no items are allocated, matrix X is all zeros, except for last column, which displays
    course capacities.

    Args:
        items (list[ScheduleItem]): Items from class BaseItem
        agents (list[BaseAgent]): Agents from class BaseAgent

    Returns:
        X: len(items) x (len(agents)+1) numpy array
    """
    n = len(items)
    m = len(agents) + 1
    X = np.zeros([n, m], dtype=int)
    for i in range(n):
        X[i][m - 1] = items[i].capacity
    return X


def initialize_edge_matrix(N: int):
    """Initialize array that keeps track of the agents responsible for a specific edge between two
    different items on the exchange graph.
    Initially, there are no edges, thus edge matrix is a len(items) x len(items) array of empty lists.

    Args:
       N (int): Number of items

    Returns:
        list: N x N list of empty lists
    """
    edge_matrix = []
    for i in range(N):
        edge_row = []
        for j in range(N):
            edge_row.append([])
        edge_matrix.append(edge_row)
    return edge_matrix


def initialize_exchange_graph(N: int):
    """Generate exchange graph. There is one node for every item and a sink node 't' representing the pile of unnasigned items.
    Initially, there are no edges between items, and an edge from every item node to node 't'.
    Disclaimer: The previous assumes that every items has capacity > 0

    Args:
        N (int): number of items

    Returns:
        nx.graph: networkx graph object
    """
    exchange_graph = nx.DiGraph()
    for i in range(N):
        exchange_graph.add_node(i)
    exchange_graph.add_node("t")
    for i in range(N):
        exchange_graph.add_edge(i, "t")
    return exchange_graph


"""Retrieve/update information"""


def get_gain_function(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    agent_picked: int,
    criteria: str,
    weights: list[float],
):
    """Get gain function for a certain agent from the current allocation, according to the gain function criteria and agent assigned weight.
    This function is used to update the gain function for the item that just played, to keep track of priority agents.

    Args:
        X (type[np.ndarray]): allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        items (list[ScheduleItem]): Items from class BaseItem
        agent_picked (int): index of the agent that just played
        criteria (str): general yankee swap criteria
        weights (list[float]): list of weights assigned to the agents, if any

    Returns:
        float: updated gain fucntion for the agent that just played
    """
    agent = agents[agent_picked]
    bundle = get_bundle_from_allocation_matrix(X, items, agent_picked)
    val = agent.valuation(bundle)
    if criteria == "LorenzDominance":
        return -val
    w_i = weights[agent_picked]
    if criteria == "WeightedLeximin":
        return -val / w_i
    if criteria == "WeightedNash":
        if val == 0:
            return float("inf")
        else:
            return (1 + 1 / val) ** w_i
    if criteria == "WeightedHarmonic":
        return w_i / (val + 1)


def get_owners_list(X: type[np.ndarray], item_index: int):
    """From the exchange matrix, list of indeces of all agents that currently have certain item.

    Args:
        X (type[np.ndarray]): Allocation matrix
        item_index (int): index of the item for which we want to get the owners

    Returns:
        list[int]: list of item's owners' indeces
    """
    item_list = X[item_index]
    owners_list = np.nonzero(item_list)
    return owners_list[0]


def get_bundle_from_allocation_matrix(
    X: type[np.ndarray], items: list[ScheduleItem], agent_index: int
):
    """Get list of all items currently owned by a certain agent (bundle), given the current allocation

    Args:
        X (type[np.ndarray]): Allocation matrix
        items (list[ScheduleItem]): List of items from class BaseItem
        agent_index (int): index of the agent for which we want to get the current bundle
    Returns:
        list[ScheduleItem]: List of items from the BaseItem class currently owned by the agent
    """
    bundle0 = []
    items_list = X[:, agent_index]
    for i in range(len(items_list)):
        if int(items_list[i]) == 1:
            bundle0.append(items[i])
    return bundle0


def get_bundle_indexes_from_allocation_matrix(X: type[np.ndarray], agent_index: int):
    """Get list of indices of all items currently owned by a certain agent (bundle), given the current allocation

    Args:
        X (type[np.ndarray]): Allocation matrix
        agent_index (int): index of the agent for which we want to get the current bundle
    Returns:
        list[int]: List of indices of the items from the BaseItem class currently owned by the agent
    """
    bundle_indexes = []
    items_list = X[:, agent_index]
    for i in range(len(items_list)):
        if int(items_list[i]) == 1:
            bundle_indexes.append(i)
    return bundle_indexes


def get_multiple_agents_desired_items(
    agents: list[BaseAgent], items: list[ScheduleItem], agents_indexes: list[int]
):
    """Get list of unique desired items from union of items desired by multiple agents

    Args:
        agents (list[BaseAgent]): Agents from class BaseAgent
        items (list[ScheduleItem]): Items from class BaseItem
        agents_indexes (list[int]): list of indices of agents

    Returns:
        list[int]: list of items indices
    """
    lis = []
    for agent_index in agents_indexes:
        agent = agents[agent_index]
        lis = lis + agent.get_desired_items_indexes(items)
    return list(set(lis))


def get_multiple_agents_bundles(X: type[np.ndarray], agents_indexes: list[int]):
    """Get list of unique items from union of items owned by multiple agents

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents_indexes (list[int]): list of indices of agents

    Returns:
        list[int]: list of items indices
    """
    lis = []
    for agent_index in agents_indexes:
        lis = lis + get_bundle_indexes_from_allocation_matrix(X, agent_index)
    return list(set(lis))


def find_agent(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    current_item_index: int,
    last_item_index: int,
):
    """Find index of an agent that is currently willing to exchange a current item for a certain other item.
    This will depend on their current bundle, for which the allocation matrix is needed.

    Args:
        X (type[np.ndarray]): allocation matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        current_item_index (int): index of the item that we want to exchange
        last_item_index (int): index of the item that we want to exchange current item for

    Returns:
        item: index of the agent williing to do the exchange
    """
    owners = get_owners_list(X, current_item_index)
    for owner in owners:
        agent = agents[owner]
        bundle = get_bundle_from_allocation_matrix(X, items, owner)
        if agent.exchange_contribution(
            bundle, items[current_item_index], items[last_item_index]
        ):
            return owner
    print(
        "Agent not found"
    )  # this should never happen. If the item was in the path, then someone must be willing to exchange it


"""Update allocation after finding the shortest path in exchange graph"""


def update_allocation(
    X: type[np.ndarray],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agent_picked: int,
):
    """Execute the transfer path found, updating the allocation of items accordingly

    Args:
        X (type[np.ndarray]): allocation matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agent_picked (int): index of the agent currently playing

    Returns:
        X (type[np.ndarray]): updated allocation matrix
        agents_involved (list[int]): indices of the agents involved in the transfer path
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    agents_involved = [agent_picked]
    X[last_item, len(agents)] -= 1
    while len(path) > 0:
        last_item = path.pop(len(path) - 1)
        # print('last item: ', last_item)
        if len(path) > 0:
            next_to_last_item = path[-1]
            current_agent = find_agent(X, agents, items, next_to_last_item, last_item)
            agents_involved.append(current_agent)
            X[last_item, current_agent] = 1
            X[next_to_last_item, current_agent] = 0
        else:
            X[last_item, agent_picked] = 1

    return X, agents_involved


def update_allocation_E(
    X: type[np.ndarray],
    G: type[nx.Graph],
    E: list[list],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agent_picked: int,
):
    """Execute the transfer path found, updating the allocation of items and edge matrix accordingly.
    Edge matrix is a list of lists containing the indices of agents responsible for each edge on the exchange graph
    This function is for the edge_matrix version of yankee swap

    Args:
        X (type[np.ndarray]): allocation matrix
        G (type[nx.Graph]): exchange graph
        E (list[list]): edge matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agent_picked (int): index of the agent currently playing

    Returns:
        X (type[np.ndarray]): updated allocation matrix
        G (type[nx.Graph]): updated exchange graph
        E (list[list]): updated edge matrix
        agents_involved (list[int]): indices of the agents involved in the transfer path
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    agents_involved = [agent_picked]
    X[last_item, len(agents)] -= 1
    while len(path) > 0:
        last_item = path.pop(len(path) - 1)
        if len(path) > 0:
            next_to_last_item = path[-1]
            current_agent = E[next_to_last_item][last_item][0]
            agents_involved.append(current_agent)
            X[last_item, current_agent] = 1
            X[next_to_last_item, current_agent] = 0
            for item_index in range(len(items)):
                if current_agent in E[next_to_last_item][item_index]:
                    E[next_to_last_item][item_index].remove(current_agent)
                    if len(E[next_to_last_item][item_index]) == 0 and G.has_edge(
                        next_to_last_item, item_index
                    ):
                        G.remove_edge(next_to_last_item, item_index)
        else:
            X[last_item, agent_picked] = 1
    return X, G, E, agents_involved


"""Graph functions for the exchange graph"""


def find_shortest_path(G: type[nx.Graph], start: str, end: str):
    """Find and return shortest path from start to end nodes on graph G. Return False if there is no path

    Args:
        G (type[nx.Graph]): exchange graph
        start (str): start node
        end (str): target node

    Returns:
        list[int]: list of nodes (item indices) on the shortest path
        of False: if there is no such path
    """
    try:
        p = nx.shortest_path(G, source=start, target=end)
        return p
    except:
        return False


def add_agent_to_exchange_graph(
    X: type[np.ndarray],
    G: type[nx.Graph],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    agent_picked: int,
):
    """Create node representing the agent currently playing, add edges from the node to items that would increase their utility

    Args:
        X (type[np.ndarray]): allocation matrix
        G (type[nx.Graph]): exchange graph
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        agent_picked (int): index of the agent currently playing

    Returns:
        G (type[nx.Graph]): Updated exchange graph
    """
    G.add_node("s")
    bundle = get_bundle_from_allocation_matrix(X, items, agent_picked)
    agent = agents[agent_picked]
    for i in agent.get_desired_items_indexes(items):
        g = items[i]
        if (
            g not in bundle
            and agents[agent_picked].marginal_contribution(bundle, g) == 1
        ):
            G.add_edge("s", i)
    return G


def update_exchange_graph(
    X: type[np.ndarray],
    G: type[nx.Graph],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agents_involved: list[int],
):
    """Given the updated allocation, path found and list of involved agents in the transfer path, update the exchange graph

    Args:
        X (type[np.ndarray]): allocation matrix
        G (type[nx.Graph]): exchange graph
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agents_involved (list[int]): list of the indices of the agents invovled in the transfer path

    Returns:
        G (type[nx.Graph]): updated exchange graph
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    if X[last_item, len(agents)] == 0:
        G.remove_edge(last_item, "t")
    agents_involved_desired_items = get_multiple_agents_desired_items(
        agents, items, agents_involved
    )
    agents_involved_bundles = get_multiple_agents_bundles(X, agents_involved)
    for item_idx in agents_involved_bundles:
        item_1 = items[item_idx]
        owners = list(get_owners_list(X, item_idx))
        if len(agents) in owners:
            owners.remove(len(agents))
        owners_desired_items = get_multiple_agents_desired_items(agents, items, owners)
        items_to_loop_over = list(
            set(agents_involved_desired_items + owners_desired_items)
        )
        for item_2_idx in items_to_loop_over:
            if item_2_idx != item_idx:
                item_2 = items[item_2_idx]
                exchangeable = False
                for owner in owners:
                    if owner != len(agents):
                        agent = agents[owner]
                        bundle_owner = get_bundle_from_allocation_matrix(
                            X, items, owner
                        )
                        willing_owner = agent.exchange_contribution(
                            bundle_owner, item_1, item_2
                        )
                        if willing_owner:
                            exchangeable = True
                            break
                if exchangeable:
                    if not G.has_edge(item_idx, item_2_idx):
                        G.add_edge(item_idx, item_2_idx)
                else:
                    if G.has_edge(item_idx, item_2_idx):
                        G.remove_edge(item_idx, item_2_idx)
    return G


def update_exchange_graph_E(
    X: type[np.ndarray],
    G: type[nx.Graph],
    E: list[list],
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    path_og: list[int],
    agents_involved: list[int],
):
    """_summary_

    Args:
        X (type[np.ndarray]): allocation matrix
        G (type[nx.Graph]): exchange graph
        E (list[list]): edge matrix
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        path_og (list[int]): shortest path, list of items indices
        agents_involved (list[int]): list of the indices of the agents invovled in the transfer path

    Returns:
        G (type[nx.Graph]): updated exchange graph
        E (list[list]): updated edge matrix
    """
    path = path_og.copy()
    path = path[1:-1]
    last_item = path[-1]
    if X[last_item, len(agents)] == 0:
        G.remove_edge(last_item, "t")
    for agent_index in agents_involved:
        agent = agents[agent_index]
        agent_bundle = get_bundle_indexes_from_allocation_matrix(X, agent_index)
        agent_bundle_items = get_bundle_from_allocation_matrix(X, items, agent_index)
        agent_desired_items = agent.get_desired_items_indexes(items)
        for item1_idx in agent_bundle:
            item1 = items[item1_idx]
            for item2_idx in agent_desired_items:
                item2 = items[item2_idx]
                if item1_idx != item2_idx:
                    if agent_index in E[item1_idx][item2_idx]:
                        if not agent.exchange_contribution(
                            agent_bundle_items, item1, item2
                        ):
                            E[item1_idx][item2_idx].remove(agent_index)
                            if len(E[item1_idx][item2_idx]) == 0 and G.has_edge(
                                item1_idx, item2_idx
                            ):
                                G.remove_edge(item1_idx, item2_idx)
                    else:
                        if agent.exchange_contribution(
                            agent_bundle_items, item1, item2
                        ):
                            E[item1_idx][item2_idx].append(agent_index)
                            if not G.has_edge(item1_idx, item2_idx):
                                G.add_edge(item1_idx, item2_idx)
    return G, E


"""Allocation algorithms"""


def SPIRE_algorithm(agents: list[BaseAgent], items: list[ScheduleItem]):
    """SPIRE allocation algorithm.
    In each round, give the playing agent all items they can add to their bundle that give them positive utility

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem

    Returns:
         X (type[np.ndarray]): allocation matrix
    """
    X = initialize_allocation_matrix(items, agents)
    agent_index = 0
    for agent_index, agent in enumerate(agents):
        bundle = []
        desired_items = agent.get_desired_items_indexes(items)
        for item in desired_items:
            if X[item, len(agents)] > 0:
                current_val = agent.valuation(bundle)
                new_bundle = bundle.copy()
                new_bundle.append(items[item])
                new_valuation = agent.valuation(new_bundle)
                if new_valuation > current_val:
                    X[item, agent_index] = 1
                    X[item, len(agents)] -= 1
                    bundle = new_bundle.copy()
    return X


def round_robin(agents: list[BaseAgent], items: list[ScheduleItem]):
    """Round Robin allocation algorithm.
    In each round, give the playing agent one item they can add to their bundle that give them positive utility, if any

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem

    Returns:
         X (type[np.ndarray]): allocation matrix
    """
    players = list(range(len(agents)))
    X = initialize_allocation_matrix(items, agents)
    while len(players) > 0:
        for player in players:
            val = 0
            current_item = []
            agent = agents[player]
            desired_items = agent.get_desired_items_indexes(items)
            bundle = get_bundle_from_allocation_matrix(X, items, player)
            for item in desired_items:
                if X[item, len(agents)] > 0:
                    current_val = agent.marginal_contribution(bundle, items[item])
                    if current_val > val:
                        current_item.clear()
                        current_item.append(item)
                        val = current_val
            if len(current_item) > 0:
                X[current_item[0], player] = 1
                X[current_item[0], len(agents)] -= 1
            else:
                players.remove(player)
    return X


def round_robin_weights(
    agents: list[BaseAgent], items: list[ScheduleItem], weights: list[float]
):
    """Round Robin allocation algorithm, considering different weights among
    In each round, give the playing agent one item they can add to their bundle that give them positive utility, if any

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        weights (list[float]): list of agents assigned weights

    Returns:
        X (type[np.ndarray]): allocation matrix
    """
    players = list(range(len(agents)))
    X = initialize_allocation_matrix(items, agents)
    weights_aux = weights.copy()
    while len(players) > 0:
        weight = weights_aux[0]
        for player in players:
            if weights[player] == weight:
                val = 0
                current_item = []
                agent = agents[player]
                desired_items = agent.get_desired_items_indexes(items)
                bundle = get_bundle_from_allocation_matrix(X, items, player)
                for item in desired_items:
                    if X[item, 0] > 0:
                        current_val = agent.marginal_contribution(bundle, items[item])
                        if current_val > val:
                            current_item.clear()
                            current_item.append(item)
                            val = current_val
                if len(current_item) > 0:
                    X[current_item[0], player] = 1
                    X[current_item[0], 0] -= 1
                else:
                    players.remove(player)
                    weights_aux.pop(0)
    return X


def general_yankee_swap(
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    criteria: str = "LorenzDominance",
    weights: list[float] = [],
    plot_exchange_graph: bool = False,
):
    """General Yankee swap allocation algorithm.

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        criteria (str, optional): gain function criteria. Defaults to "LorenzDominance". See get_gain_function to see other alternatives
        weights (list[float]): list of agents assigned weights
        plot_exchange_graph (bool, optional): Defaults to False. Change to True to display exchange graph plot after every modification to it.

    Returns:
        X (type[np.ndarray]): allocation matrix
        time_steps (list[float]): time elapsed until the end of every iteration
        agents_involved_arr (list[int]): nuber of agents involved in every iteration
    """
    N = len(items)
    M = len(agents)
    players = list(range(M))
    X = initialize_allocation_matrix(items, agents)
    G = initialize_exchange_graph(N)
    gain_vector = np.zeros([M])
    count = 0
    time_steps = []
    agents_involved_arr = []
    start = time.process_time()
    while len(players) > 0:
        print("Iteration: %d" % count, end="\r")
        count += 1
        agent_picked = np.argmax(gain_vector)
        G = add_agent_to_exchange_graph(X, G, agents, items, agent_picked)
        if plot_exchange_graph:
            nx.draw(G, with_labels=True)
            plt.show()

        path = find_shortest_path(G, "s", "t")
        G.remove_node("s")

        if path == False:
            players.remove(agent_picked)
            gain_vector[agent_picked] = float("-inf")
            time_steps.append(time.process_time() - start)
            agents_involved_arr.append(0)
        else:
            X, agents_involved = update_allocation(X, agents, items, path, agent_picked)
            G = update_exchange_graph(X, G, agents, items, path, agents_involved)
            gain_vector[agent_picked] = get_gain_function(
                X, agents, items, agent_picked, criteria, weights
            )
            if plot_exchange_graph:
                nx.draw(G, with_labels=True)
                plt.show()
            time_steps.append(time.process_time() - start)
            agents_involved_arr.append(len(agents_involved))
    return X, time_steps, agents_involved_arr


def general_yankee_swap_E(
    agents: list[BaseAgent],
    items: list[ScheduleItem],
    criteria: str = "LorenzDominance",
    weights: list = [],
    plot_exchange_graph: bool = False,
):
    """General Yankee swap allocation algorithm, edge matrix version
    Equivalent to general_yankee_swap, just different book keeping to speed things up

    Args:
        agents (list[BaseAgent]): List of agents from class BaseAgent
        items (list[ScheduleItem]): List of items from class BaseItem
        criteria (str, optional): gain function criteria. Defaults to "LorenzDominance". See get_gain_function to see other alternatives
        weights (list[float]): list of agents assigned weights
        plot_exchange_graph (bool, optional): Defaults to False. Change to True to display exchange graph plot after every modification to it.

    Returns:
        X (type[np.ndarray]): allocation matrix
        time_steps (list[float]): time elapsed until the end of every iteration
        agents_involved_arr (list[int]): nuber of agents involved in every iteration
    """
    N = len(items)
    M = len(agents)
    players = list(range(M))
    X = initialize_allocation_matrix(items, agents)
    G = initialize_exchange_graph(N)
    E = initialize_edge_matrix(N)
    gain_vector = np.zeros([M])
    count = 0
    time_steps = []
    agents_involved_arr = []
    start = time.process_time()
    while len(players) > 0:
        print("Iteration: %d" % count, end="\r")
        count += 1
        agent_picked = np.argmax(gain_vector)
        G = add_agent_to_exchange_graph(X, G, agents, items, agent_picked)
        if plot_exchange_graph:
            nx.draw(G, with_labels=True)
            plt.show()

        path = find_shortest_path(G, "s", "t")
        G.remove_node("s")

        if path == False:
            players.remove(agent_picked)
            gain_vector[agent_picked] = float("-inf")
            time_steps.append(time.process_time() - start)
            agents_involved_arr.append(0)
        else:
            X, G, E, agents_involved = update_allocation_E(
                X, G, E, agents, items, path, agent_picked
            )
            G, E = update_exchange_graph_E(
                X, G, E, agents, items, path, agents_involved
            )
            gain_vector[agent_picked] = get_gain_function(
                X, agents, items, agent_picked, criteria, weights
            )
            if plot_exchange_graph:
                nx.draw(G, with_labels=True)
                plt.show()
            time_steps.append(time.process_time() - start)
            agents_involved_arr.append(len(agents_involved))
    return X, time_steps, agents_involved_arr
