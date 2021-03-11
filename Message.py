import numpy as np


class Message:
    pass


if __name__ == "__main__":
    msg1 = np.random.rand(500, 1)
    np.save("./data1.npy", msg1)
    msg2 = np.random.rand(500, 1)
    np.save("./data2.npy", msg2)
