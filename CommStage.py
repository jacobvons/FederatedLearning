import enum


class CommStage(enum.Enum):
    CONN_ESTAB = 0
    PARAM_DIST = 1
    REPORT = 2
    END = 3
