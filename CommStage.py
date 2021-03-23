import enum


class CommStage(enum.Enum):
    CONN_ESTAB = 0
    PC_INFO_EXCHANGE = 1
    PC_AGGREGATION = 2
    PARAM_DIST = 3
    REPORT = 4
    END = 5
