import numpy as np


def recv_large(sock):
    """
    Receive a large formatted message from a socket and return it

    :param sock: the socket used for receiving
    :return: raw bytes message with the bytes ending b"end" stripped
    """
    data = b""
    while True:
        pack = sock.recv(10)
        data += pack
        if data[-3:] == b"end":
            break
    return data[:-3]


def format_msg(bytes_msg):
    """
    Formatting a bytes message with a b"end" ending

    :param bytes_msg: the bytes message that needs formatting
    :return: bytes message with b"end" at the end
    """
    return bytes_msg + b"end"


def stringify(arr):
    """
    Turn a numpy array into a structured string for encryption

    :param arr: numpy array
    :return: a structured byte string representing the original array
    """
    arr_str = ""
    for j, row in enumerate(arr):
        try:
            for i, entry in enumerate(row):
                arr_str += str(entry)
                arr_str += "_" if i != len(row) - 1 else "#"
        except TypeError:
            arr_str += str(row) + "_" if j != len(arr) - 1 else str(row) + "#"
    return arr_str.encode("utf-8")


def destringify(b_str, is_bias=False):
    """
    Turn a byte string back into the numpy array it represents

    :param b_str: a byte string
    :return: the numpy array the byte string represents
    """
    raw_str = b_str.decode("utf-8")
    rows = raw_str.split("#")
    arr = []
    for row in rows:
        if not row:
            break
        entries = row.split("_")
        float_row = list(map(float, entries))
        arr.append(float_row)
    arr = np.array(arr, dtype="float64")
    if is_bias:
        arr = arr[0]
    return arr
