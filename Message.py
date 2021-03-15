import numpy as np
import torch
from sklearn.neural_network import MLPRegressor


class Message:

    def __init__(self, message, comm_stage):
        self.message = message
        self.comm_stage = comm_stage


if __name__ == "__main__":
    msg1 = np.random.rand(50, 1)
    np.save("./data1.npy", msg1)
    msg2 = np.random.rand(50, 1)
    np.save("./data2.npy", msg2)
    init_msg = np.random.rand(50, 1)
    np.save("./init_data.npy", init_msg)

    # PyTorch model
    init_model = torch.nn.Linear(50, 1)
    torch.save(init_model, "init_model.pth")
    print(init_model.weight)
    print(init_model.bias)

    # sklearn model
    # init_model = MLPRegressor(hidden_layer_sizes=(50, ), activation="tanh", solver="adam",
    #                           alpha=0.0001, batch_size="auto", learning_rate="constant",
    #                           learning_rate_init=0.001, max_iter=5, shuffle=True,
    #                           random_state=42, momentum=0.9, early_stopping=False,
    #                           validation_fraction=0.15)
    # Need to init MLPRegressor by calling fit() on some random data
    # print(init_model.coefs_)
    # print(init_model.t_)

    dummy_msg = np.array([0])
    np.save("./dummy_data.npy", dummy_msg)
