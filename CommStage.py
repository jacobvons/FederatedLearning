import enum


class CommStage(enum.Enum):
    """
    Communication stages in the pipeline.

    CONN_ESTAB: connection establishment stage
    PC_INFO_EXCHANGE: principle component meta-information exchange stage
    PC_AGGREGATION: principle component aggregation stage
    PARAM_DIST: parameter distribution stage
    REPORT: report stage
    END: end stage
    """
    CONN_ESTAB = 0
    PC_INFO_EXCHANGE = 1
    PC_AGGREGATION = 2
    PARAM_DIST = 3
    REPORT = 4
    END = 5
