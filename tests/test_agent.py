from agent import exchange_contribution, marginal_contribution
from agent.item import ScheduleItem
from agent.valuation import ConstraintSatifactionValuation


def test_exchange_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert (
        exchange_contribution(
            course_valuation, [all_items[0]], all_items[0], all_items[1]
        )
        == True
    )

    assert (
        exchange_contribution(
            course_valuation, [all_items[0], all_items[2]], all_items[2], all_items[1]
        )
        == False
    )


def test_marginal_contribution(
    course_valuation: ConstraintSatifactionValuation, all_items: list[ScheduleItem]
):
    assert marginal_contribution(course_valuation, [all_items[0]], all_items[1]) == 0
    assert marginal_contribution(course_valuation, [all_items[0]], all_items[2]) == 1
