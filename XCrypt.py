import numpy as np


def xcrypt_2d(xcrypt, message):
    m, n = message.shape
    output = []
    for i in range(0, m):
        output.append(np.array(list(map(xcrypt, message[i]))))
    return np.array(output)