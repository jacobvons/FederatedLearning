import numpy as np


class Message:
    pass


if __name__ == "__main__":
    msg = np.random.rand(500, 1)
    np.save("./data1.npy", msg)
