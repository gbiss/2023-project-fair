import os
import random
from itertools import chain, combinations

import pandas as pd

NUM_ROWS = 10

FILE_PATH = os.path.join(os.path.dirname(__file__), "../resources/random_survey.csv")
QUESTION_HEADER = "1,2,3,4,5"
CICS_HEADER = "7 _1,7 _2,7 _3,7 _4,7 _5,7 _6,7 _7,7 _8,7 _9,7 _10,7 _11,7 _12,7 _13,7 _14,7 _15,7 _16,7 _17,7 _18,7 _19,7 _20,7 _21,7 _22,7 _23,7 _24,7 _25,7 _26,7 _27,7 _28,7 _29,7 _30".split(
    ","
)
COMPSCI_HEADER = "7_1,7_2,7_3,7_4,7_5,7_6,7_7,7_8,7_9,7_10,7_11,7_12,7_13,7_14,7_15,7_16,7_17,7_18,7_19,7_20,7_21,7_22,7_23,7_24,7_25,7_26,7_27,7_28,7_29,7_30,7_31,7_32,7_33,7_34,7_35,7_36,7_37,7_38,7_39,7_40,7_41,7_42,7_43,7_44,7_45,7_46,7_47,7_48,7_49,7_50,7_51,7_52,7_53,7_54,7_55,7_56,7_57,7_58,7_59,7_60,7_61,7_62,7_63,7_64,7_65,7_66,7_67,7_68,7_69,7_70,7_71".split(
    ","
)
INFO_HEADER = "7 _1.1,7 _2.1,7 _3.1,7 _4.1,7 _5.1,7 _6.1,7 _7.1".split(",")


def powerset(iter: range) -> list:
    """Generate the power set

    Args:
        iter (range): Items from which to generate the power set

    Returns:
        list: A list of all sets in power set
    """
    s = list(iter)

    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def random_questions() -> pd.Series:
    """Randomly generate responses for the first five questions

    Returns:
        pd.Series: Responses to questions
    """
    q1 = random.choice(
        [
            "Freshman",
            "Sophomore",
            "Junion",
            "Senior",
            "MS Student",
            "MS/PhD or PhD Student",
        ]
    )
    q2 = random.randint(0, 7)
    q3 = random.randint(0, 7)
    q4 = ",".join([str(i) for i in random.choice(powerset(range(1, 5)))])
    columns = ["1", "2", "3", "4"]
    q5 = []
    for i in range(1, 12):
        q5 += [",".join([str(i) for i in random.choice(powerset(range(1, 6)))])]
        columns.append(f"5#1_{i}")

    return pd.Series(data=[q1, q2, q3, q4] + q5, index=columns)


def random_row() -> pd.Series:
    """Randomly generate a single row of the survey

    Returns:
        pd.Series: A Series representing the row
    """
    questions = random_questions()
    cics = pd.Series(
        data=[random.randint(1, 8) for i in range(len(CICS_HEADER))], index=CICS_HEADER
    )
    compsci = pd.Series(
        data=[random.randint(1, 8) for i in range(len(COMPSCI_HEADER))],
        index=COMPSCI_HEADER,
    )
    info = pd.Series(
        data=[random.randint(1, 8) for i in range(len(INFO_HEADER))], index=INFO_HEADER
    )

    return pd.concat([questions, cics, compsci, info])


pd.DataFrame([random_row() for i in range(NUM_ROWS)]).to_csv(FILE_PATH, index=False)
