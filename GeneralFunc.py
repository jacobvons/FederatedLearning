def recv_large(sock):
    data = b""
    while True:
        pack = sock.recv(1024)
        data += pack
        if data[-3:] == b"end":
            break
    return data[:-3]


def format_msg(binary_msg):
    return binary_msg + b"end"
