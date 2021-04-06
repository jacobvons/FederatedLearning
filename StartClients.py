import os
import torch
from threading import Thread
from Client import *


def release_client(client_id, host, port, path):
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
