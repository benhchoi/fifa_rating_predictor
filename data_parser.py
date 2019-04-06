import csv
import collections
import codecs
from enum import IntEnum


class Data(IntEnum):
    ROW = 0
    ID = 1
    NAME = 2
    AGE = 3
    PHOTO = 4  # url
    NATIONALITY = 5
    FLAG_PIC = 6  # url
    OVERALL = 7  # /100
    POTENTIAL = 8  # /100
    CLUB = 9
    CLUB_LOGO = 10  # enum
    VALUE = 11
    WAGE = 12
    SPECIAL = 13  # some sum of various skill stats
    PREF_FOOT = 14  # right or left
    REPUTATION = 15  # /5 START
    WEAK_FOOT = 16
    SKILL_MOVES = 17  # /5 END
    WORK_RATE = 18
    BODY_TYPE = 19
    REAL_FACE = 20  # bool
    POSITION = 21
    JERSEY_NUM = 22
    JOINED = 23
    LOANED_FROM = 24
    CONTRACT_END = 25
    HEIGHT = 26
    WEIGHT = 27
    LS = 28  # /100 start, but with strings
    ST = 29
    RS = 30
    LW = 31
    LF = 32
    CF = 33
    RF = 34
    RW = 35
    LAM = 36
    CAM = 37
    RAM = 38
    LM = 39
    LCM = 40
    CM = 41
    RCM = 42
    RM = 43
    LWB = 44
    LDM = 45
    CDM = 46
    RDM = 47
    RWB = 48
    LB = 49
    LCB = 50
    CB = 51
    RCB = 52
    RB = 53  # /100 end, but with strings
    CROSSING = 54  # /100 start, no strings
    FINISHING = 55
    HEADING_ACC = 56
    SHORT_PASS = 57
    VOLLEYS = 58
    DRIBBLING = 59
    CURVE = 60
    FK_ACC = 61
    LONG_PASS = 62
    BALL_CONTROL = 63
    ACCEL = 64
    SPRINT_SPEED = 65
    AGILITY = 66
    REACTIONS = 67
    BALANCE = 68
    SHOT_POWER = 69
    JUMPING = 70
    STAMINA = 71
    STRENGTH = 72
    LONG_SHOTS = 73
    AGGRESSION = 74
    INTERCEPTIONS = 75
    POSITIONING = 76
    VISION = 77
    PENALTIES = 78
    COMPOSURE = 79
    MARKING = 80
    STAND_TACKLE = 81
    SLIDE_TACKLE = 82
    GK_DIVING = 83
    GK_HANDLING = 84
    GK_KICKING = 85
    GK_POSITIONING = 86
    GK_REFLEXES = 87  # /100 end, no strings
    REL_CLAUSE = 88  # monetary value


def get_csv(filepath):
    with codecs.open(filepath, 'r', encoding="utf-8", errors="ignore") \
            as filestring:
        return [row for count, row in enumerate(csv.reader(filestring))
                if count != 0]


# takes in data that is all string values and converts them to their original
# data types
def convert_data(data):
    all_num_cols = {Data.ROW, Data.ID, Data.AGE, Data.OVERALL, Data.POTENTIAL,
                    Data.SPECIAL, Data.REPUTATION, Data.WEAK_FOOT,
                    Data.SKILL_MOVES, Data.JERSEY_NUM}
    all_num_cols = {int(elem) for elem in all_num_cols}

    for i in range(54, 88):
        all_num_cols.add(i)

    converted_data = []
    for row in data:
        converted_row = []
        for col, value in enumerate(row):
            if col in all_num_cols:
                if value:
                    converted_row.append(int(value))
                else:
                    converted_row.append(0)
            else:
                converted_row.append(value)
        converted_data.append(converted_row)

    return converted_data
