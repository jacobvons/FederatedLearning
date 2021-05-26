import subprocess
from threading import Thread
from ArgReader import *


def parse_args(args: dict) -> list:
    parsed = []
    for k, v in args.items():
        parsed.append("--"+str(k))
        parsed.append(str(v))
    return parsed


def create_fed(args: dict):  # TODO: Instead of doing the whole thing, create a Federator first
    cmd = ["python", "Federator.py"] + parse_args(args)
    subprocess.call(cmd)


def start_clients():
    subprocess.call(["python", "StartClients.py"])


if __name__ == "__main__":
    reader = ArgReader("./test_args.csv")
    reader.parse()
    fed_args = reader.args
    for index, arg_dict in enumerate(fed_args):
        fed_thread = Thread(target=create_fed, args=(arg_dict,))
        clients_thread = Thread(target=start_clients)
        fed_thread.start()
        clients_thread.start()
        fed_thread.join()
        clients_thread.join()
