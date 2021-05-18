import subprocess
from threading import Thread


def parse_args(args: dict) -> list:
    l = []
    for k, v in args.items():
        l.append("--"+k)
        l.append(str(v))
    return l


def create_fed(args: dict):
    l = ["python", "Federator.py"] + parse_args(args)
    subprocess.call(l)


def start_clients():
    subprocess.call(["python", "StartClients.py"])


if __name__ == "__main__":
    fed_args = {"h": "127.0.0.1", "p": 65432, "n": 8, "rounds": 2, "ratio": 0.85, "x": 1, "e": 3, "name": "test"}
    fed_thread = Thread(target=create_fed, args=(fed_args, ))
    clients_thread = Thread(target=start_clients)

    fed_thread.start()
    clients_thread.start()

    fed_thread.join()
    clients_thread.join()
