import numpy as np

from .agent import BaseAgent, LegacyStudent
from .allocation import get_bundle_from_allocation_matrix, general_yankee_swap_E
from .constraint import CourseTimeConstraint, MutualExclusivityConstraint
from .feature import Course, Section
from .item import ScheduleItem
from .simulation import SubStudent


def utilitarian_welfare(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute utilitarian social welfare (USW)

    Calculates the average of utilities across all agents.

    Args:
        X (type[np.ndarray]): Allocation matrix
        agents (list[BaseAgent]): Agents from class BaseAgent
        schedule (list[ScheduleItem]): Items from class BaseItem

    Returns:
        float: USW / len(agents)
    """
    util = 0
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        util += val
    return util / (len(agents))


def nash_welfare(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute Nash social welfare (NSW)

    Calculates the number of agents with 0 utility and product of utilities across all
    agents with utility > 0.

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
    for agent_index, agent in enumerate(agents):
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
    for agent_index, agent in enumerate(agents):
        bundle = get_bundle_from_allocation_matrix(X, items, agent_index)
        val = agent.valuation(bundle)
        valuations.append(val)
    valuations.sort()
    valuations.reverse()
    return valuations


##ENVY METRICS


def EF_count(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute envy count

    Compare every agent to all other agents, add 1 to envy count if the agent gets higher
    utlity from the other agents bundle.
    NOTE: we can add 1 multiple times for the same agent if they envy multiple other agents.

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
                    envy_count += 1
    return envy_count


def EF_agents(X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]):
    """Compute envy agents count

    Compare every agent to all other agents, add 1 to envy count if the agent gets higher
    utlity from another agent's bundle.
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
                    envy_count += 1
                    break
    return envy_count


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


# FUNCTIONS TO COMPUTE PAIRWISE MAXIMIN SHARE


def create_sub_schedule(bundle_1: list[ScheduleItem], bundle_2: list[ScheduleItem]):
    """Given two subsets of the set of items, create a new sub schedule considering the items in the union of both sets.
    Capacities of the new schedule are determined by repetitions of the items through both bundles.
    This function is necessary to compute PMMS metric. Creates a schedule with the items currently own by two agents, in order to solve subproblem.

    Args:
        bundle_1 (list[ScheduleItem]): Items from class BaseItem
        bundle_2 (list[ScheduleItem]): Items from class BaseItem

    Returns:
        new_schedule (list[ScheduleItem]): Items from class BaseItem, new reduced schedule
        course_strings (list[str]): List of course strings of the new schedule
        course (type[Course]): Course instance of the new schedule
    """
    sub_schedule = [*bundle_1, *bundle_2]
    set_sub_schedule = sorted(list(set(sub_schedule)), key=lambda item: item.values[0])

    course_strings = sorted([item.values[0] for item in set_sub_schedule])
    course = Course(course_strings)
    section = Section(sorted(list(set([item.values[3] for item in set_sub_schedule]))))
    slot = sub_schedule[0].features[1]
    weekday = sub_schedule[0].features[2]

    features = [course, slot, weekday, section]

    new_schedule = []
    for i, item in enumerate(set_sub_schedule):
        new_schedule.append(
            ScheduleItem(
                features, item.values, index=i, capacity=sub_schedule.count(item)
            )
        )
    return new_schedule, course_strings, course


def yankee_swap_sub_problem(
    agent: type[BaseAgent],
    new_schedule: list[ScheduleItem],
    course_strings: list[str],
    course: type[Course],
):
    """Given an agent and information of a reduced schedule (new_schedule, course_strings, course), compute their MMS for the reduced problem,
    considering 2 identical agents competing for the items in the reduced schedule.
    We do this by computing a leximin allocation through yankee swap.

    Args:
        agent (type[BaseAgent]): Agent from the class BaseAgent
        new_schedule (list[ScheduleItem]): Items from class BaseItem, new reduced schedule
        course_strings (list[str]): List of course strings of the new schedule
        course (type[Course]): Course instance of the new schedule

    Returns:
        int: Agent's MMS for the subproblem
    """
    course_time_constr = CourseTimeConstraint.from_items(
        new_schedule, new_schedule[0].features[1], new_schedule[0].features[2]
    )
    course_sect_constr = MutualExclusivityConstraint.from_items(new_schedule, course)
    preferred = agent.preferred_courses
    new_student = SubStudent(
        agent.student.quantities,
        [
            [item for item in pref if item in course_strings]
            for pref in agent.student.preferred_topics
        ],
        list(set(course_strings) & set(preferred)),
        agent.student.total_courses,
        course,
        [course_time_constr, course_sect_constr],
        new_schedule,
    )
    legacy_student = LegacyStudent(new_student, new_student.preferred_courses, course)
    legacy_student.student.valuation.valuation = (
        legacy_student.student.valuation.compile()
    )
    sub_student = legacy_student

    X_sub, _, _ = general_yankee_swap_E([sub_student, sub_student], new_schedule)

    bundle_1 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 0)
    bundle_2 = get_bundle_from_allocation_matrix(X_sub, new_schedule, 1)

    return min([sub_student.valuation(bundle_1), sub_student.valuation(bundle_2)])


def pairwise_maximin_share(
    agent1: type[BaseAgent],
    agent2: type[BaseAgent],
    current_bundle_1: list[ScheduleItem],
    current_bundle_2: list[ScheduleItem],
):
    """Given two agents and their current bundles, compute their Pairwise Maximin Share (PMMS)

    Args:
        agent1 (type[BaseAgent]): First agent
        agent2 (type[BaseAgent]): Second agent
        current_bundle_1 (list[ScheduleItem]): first agent's current bundle
        current_bundle_2 (list[ScheduleItem]): second agent's current bundle

    Returns:
        PMMS[BaseAgent] (type[int]): for agents 1 and 2, return their PMMS for the subproblem
    """

    PMMS = {}

    new_schedule, course_strings, course = create_sub_schedule(
        current_bundle_1, current_bundle_2
    )

    PMMS[agent1] = yankee_swap_sub_problem(agent1, new_schedule, course_strings, course)
    PMMS[agent2] = yankee_swap_sub_problem(agent2, new_schedule, course_strings, course)

    return PMMS


def PMMS_violations(
    X: type[np.ndarray], agents: list[BaseAgent], items: list[ScheduleItem]
):
    """Compute number of violations of the Pairwise Maximin Share (PMMS) for an allocation X

    Compare every agent to all agents of higher index and determine whether they receive their PMMS.
    Runs intermediate functions to compute PMMS and returns tuple with the number of comparison which did not comply with the PMMS,
    and the number of agents that for at least one comparison, did not receive their PMMS.

     Args:
         X (type[np.ndarray]): Allocation matrix
         agents (list[BaseAgent]): Agents from class BaseAgent
         schedule (list[ScheduleItem]): Items from class BaseItem

     Returns:
         int: Number of PMMS violations
         int: Number of agents who did not receive their PMMS in every comparison
    """
    PMMS_matrix = np.zeros((len(agents), len(agents)))
    for i, student_1 in enumerate(agents):
        bundle_1 = get_bundle_from_allocation_matrix(X, items, i)

        for j in range(i + 1, len(agents)):
            student_2 = agents[j]
            bundle_2 = get_bundle_from_allocation_matrix(X, items, j)

            if len(bundle_1) == 0 and len(bundle_2) == 0:
                continue

            PMMS = pairwise_maximin_share(student_1, student_2, bundle_1, bundle_2)
            PMMS_matrix[i, j] = student_1.valuation(bundle_1) - PMMS[student_1]
            PMMS_matrix[j, i] = student_2.valuation(bundle_2) - PMMS[student_2]

    return np.sum(PMMS_matrix < 0), np.sum(np.any(PMMS_matrix < 0, axis=1))
