from threading import Thread
from Client import *
import argparse


def release_client(client_id, host, port, path):
    """
    Release a Client task to the thread pool

    :param client_id: int, id of a client
    :param host: str, address of the Federator
    :param port: int, port of the Federator
    :param path: str, relative path to the .csv dataset file (NOTE: This is buggy for Windows system currently)
    :return:
    """
    client = Client(client_id, host, port, path)
    client.connect()
    client.work()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--p")
    args = parser.parse_args()

    host = "127.0.0.1"
    port = int(args.p)
    torch.set_default_dtype(torch.float64)
    # file_dir = "../dataset/"
    # file_dir = "../non_iid_sets/dup_non_iid_3/"
    file_dir = "../shuffled_sets/shuffled_03_sets/dup_shuffle_1"
    files = [f for f in os.listdir(file_dir) if f.endswith(".csv")]
    for i in range(len(files)):
        thread = Thread(target=release_client, args=(i, host, port, os.path.join(file_dir, files[i])))
        thread.start()
