import numpy as np


def xcrypt_2d(method, message):
    """
    Encrypt or decrypt a 2D numpy array of dtype float64.

    :param method: Either pk.encrypt or sk.decrypt
    :param message: 2D numpy array of shape (a, b) or (a, ) with dtype float64
    :return: Encrypted or decrypted numpy array
    """
    try:
        m, n = message.shape
    except ValueError:
        m, n = message.shape[0], 1
    output = []
    print(m, n)
    print(message)
    if n != 1:  # For encrypting a matrix
        for i in range(0, m):
            output.append(np.array(list(map(method, message[i]))))
    else:  # For encrypting a vector
        for i in range(0, m):
            try:
                output.append(np.array(list(map(method, message))))
            except:
                output.append(np.array(list(map(method, message[i]))))
    return np.array(output)
