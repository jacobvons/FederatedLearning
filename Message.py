class Message:

    def __init__(self, message, comm_stage):
        """
        Initialising a Message instance

        :param message: a Python object as the actual message body
        :param comm_stage: the communication stage indicator
        """
        self.message = message
        self.comm_stage = comm_stage
