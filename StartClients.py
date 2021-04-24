import os
import torch
from threading import Thread
from Client import *


def release_client(client_id, host, port, path):
    """
    Release a Client task to the thread pool

    :param client_id: int, id of a client
    :param host: str, address of the Federator
    :param port: int, port of the Federator
    :param path: str, relative path to the .csv dataset file (NOTE: This is buggy for Windows system currently)
    :return:
    """
    client = Client(client_id, host, port)
    client.connect()
    client.work(path)


if __name__ == "__main__":
    host, port = "127.0.0.1", 65432
    torch.set_default_dtype(torch.float64)
    file_dir = "../dataset/"
    files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    for i in range(len(files)):
        thread = Thread(target=release_client, args=(i, host, port, file_dir+files[i]))
        thread.start()
