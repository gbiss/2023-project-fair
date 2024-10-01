import numpy as np

from fair.feature import Course
from fair.item import ScheduleItem
from fair.simulation import RenaissanceMan
from fair.stats.survey import Corpus, SingleTopicSurvey


def test_single_topic_survey(
    simple_schedule: list[ScheduleItem], student: RenaissanceMan, course: Course
):
    survey = SingleTopicSurvey.from_student(simple_schedule, student, 0, 1)
    preference_mask = [
        item.value(course) in student.preferred_courses for item in simple_schedule
    ]

    assert survey.limit == student.total_courses
    assert [survey.response_map[item] for item in simple_schedule] == preference_mask
    assert np.array_equal(survey.data().flatten(), np.array(preference_mask))


def test_corpus_validation(
    simple_schedule: list[ScheduleItem],
    simple_schedule2: list[ScheduleItem],
    student: RenaissanceMan,
    student2: RenaissanceMan,
    student3: RenaissanceMan,
):
    survey1 = SingleTopicSurvey.from_student(simple_schedule, student, 0, 1)
    survey2 = SingleTopicSurvey.from_student(simple_schedule, student2, 0, 1)
    survey3 = SingleTopicSurvey.from_student(simple_schedule2, student3, 0, 1)
    corpus1 = Corpus([survey1, survey2])
    corpus2 = Corpus([survey1, survey3])

    assert corpus1._valid()
    assert not corpus2._valid()
