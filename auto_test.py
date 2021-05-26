import subprocess
from threading import Thread
from ArgReader import *
from Federator import *
import pickle


def parse_args(args: dict) -> list:
    parsed = []
    for k, v in args.items():
        parsed.append("--"+str(k))
        parsed.append(str(v))
    return parsed


def save_pickle(obj, file_name):
    with open(file_name, "wb") as out:
        pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)
    return file_name


def create_fed_instance(args: dict):
    fed = Federator(host=args["h"],
                    port=int(args["p"]),
                    client_num=int(args["n"]),
                    comm_rounds=int(args["rounds"]),
                    explain_ratio=float(args["ratio"]),
                    xcrypt=bool(args["x"]),
                    epoch_num=int(args["e"]),
                    name=args["name"])

    return fed


def fed_create(args: dict, file_name):
    fed = create_fed_instance(args)
    global FILE
    FILE = save_pickle(fed, file_name)


def fed_listen(file_name):
    subprocess.call(["python", "FedPipeline.py", "--f", file_name])


def start_clients():
    subprocess.call(["python", "StartClients.py"])


if __name__ == "__main__":
    reader = ArgReader("./test_args.csv")
    reader.parse()
    fed_args = reader.args
    FILE = None
    for index, arg_dict in enumerate(fed_args):
        fed_create_thread = Thread(target=fed_create, args=(arg_dict, index))
        print("Creating Federator instance")
        fed_create_thread.start()
        fed_create_thread.join()
        print("Created")

        fed_listen_thread = Thread(target=fed_listen, args=(FILE, ))
        clients_thread = Thread(target=start_clients)

        fed_listen_thread.start()
        clients_thread.start()

        fed_listen_thread.join()
        clients_thread.join()
