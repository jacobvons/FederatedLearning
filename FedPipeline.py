import argparse
import pickle
import torch
from threading import Thread
import socket
from CommStage import CommStage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f")
    args = parser.parse_args()
    file_name = args.f
    fed = pickle.load(file_name)
    torch.set_default_dtype(torch.float64)

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
