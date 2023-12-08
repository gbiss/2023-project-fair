import pandas as pd

from fair.feature import Course, Slot, slot_list, slots_for_time_range


def test_course():
    hash(Course(["250", "301", "611"]))


def test_slot_from_range(excel_schedule_path: str):
    with open(excel_schedule_path, "rb") as fd:
        df = pd.read_excel(fd)

    time_ranges = df["Mtg Time"].dropna().unique()
    slot = Slot.from_time_ranges(time_ranges, "15T")
