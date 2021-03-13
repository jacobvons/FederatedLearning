import numpy as np


class Message:
    pass


if __name__ == "__main__":
    msg1 = np.random.rand(50, 1)
    np.save("./data1.npy", msg1)
    msg2 = np.random.rand(50, 1)
    np.save("./data2.npy", msg2)
    init_msg = np.random.rand(50, 1)
    np.save("./init_data.npy", init_msg)
    dummy_msg = np.array([0])
    np.save("./dummy_data.npy", dummy_msg)
