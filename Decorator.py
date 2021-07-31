import functools
from ArgReader import *
import os


def hyper_tune(func):
    """
    Tune hyper parameters of a training function
    :param func: training function
    :return:
    """
    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        # config_reader = ArgReader("./hyper_configs.csv")
        # config_reader.parse()
        # for config in config_reader.args:
        #     lr_change = config["lr"]
        #     if os.path.exists(self.checkpoint_dir):
        #         pass

        return func(self, *args, **kwargs)
    return wrapped
