import subprocess
from threading import Thread
from ArgReader import *
import socket
from CommStage import CommStage
from Federator import *
import torch
import os


def parse_args(args: dict) -> list:
    """
    Converting the argument dictionary into a list to read from

    :param args: argument dictionary
    :return: parsed list of arguments
    """
    parsed = []
    for k, v in args.items():
        parsed.append("--"+str(k))
        parsed.append(str(v))
    return parsed


def create_fed(args: dict):
    """
    Creating a Federator object according to argument sets

    :param args: argument dictionary
    :return:
    """
    fed = Federator(host=args["h"],
                    port=int(args["p"]),
                    client_num=int(args["n"]),
                    comm_rounds=int(args["rounds"]),
                    explain_ratio=float(args["ratio"]),
                    pc_agg_method=args["pc_agg"],
                    model_agg_method=args["mod_agg"],
                    xcrypt=bool(args["x"]),
                    epoch_num=int(args["e"]),
                    name=args["name"])
    return fed


def fed_job(fed):
    """
    Defining a job for federator to start

    :param fed: the Federator object
    :return:
    """
    while True:
        if len(fed.all_sockets) < fed.client_num and fed.state == CommStage.CONN_ESTAB:  # Collecting client connections
            fed.sock.settimeout(0.00001)  # 0.00001s timeout for "refreshing"
            try:
                conn, addr = fed.sock.accept()  # Accept new connections
                print("Accepted connection from", addr)
                conn.setblocking(True)
                thread = Thread(target=fed.single_proceed, args=(conn, ))
                thread.start()  # Only for establishing connection
            except socket.timeout:
                continue
        elif len(fed.conns) == fed.client_num:
            fed.sock.settimeout(None)
            thread = Thread(target=fed.batch_proceed)
            thread.start()
            thread.join()  # Do NOT accept new connections until this thread finishes

        if fed.terminate:
            print(f"Shutting down Federator: {fed.name}")
            break


def start_clients(port):
    """
    Creating a subprocess for the clients to run

    :param port: port the federator host listens on
    :return:
    """
    subprocess.call(["python", "StartClients.py", "--p", str(port)])


if __name__ == "__main__":
    reader = ArgReader("./test_args.csv")
    reader.parse()
    fed_args = reader.args
    torch.set_default_dtype(torch.float64)
    for index, arg_dict in enumerate(fed_args):
        start = time.time()
        fed = create_fed(arg_dict)
        fed_thread = Thread(target=fed_job, args=(fed, ))
        clients_thread = Thread(target=start_clients, args=(arg_dict["p"], ))
        fed_thread.start()
        clients_thread.start()
        fed_thread.join()
        clients_thread.join()
        end = time.time()
        with open("./time_records.txt", "a") as f:
            f.write(f"{arg_dict['name']}: {end-start}\n")
