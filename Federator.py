import socket
import selectors
import types
import sys
import pickle
import numpy as np


class Federator:

    def __init__(self, host, port, client_num):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        print('listening on', (host, port))
        self.sock.setblocking(False)
        self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        self.all_data = {}
        self.all_sockets = {}
        self.client_num = client_num

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()
        print("Accepted connection from", addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"", id="")
        event_actions = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, event_actions, data=data)


if __name__ == "__main__":
    host, port, client_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fed = Federator(host, port, client_num)

    while True:
        # Aggregation when data from all clients are present
        if len(fed.all_data) == client_num:
            print("Aggregating...")
            # Turn bytes info of data.outb into actual arrays
            for data in fed.all_data.values():
                data.outb = pickle.loads(data.outb)
            # Sum them and update all
            new_info = sum([data.outb for data in fed.all_data.values()])

            # Send out new parameters after aggregation
            print("Distributing new model...")
            for client_id, data in fed.all_data.items():
                sock = fed.all_sockets[client_id]
                print('Sending information to', data.addr, "(client", data.id + ")...")
                data.outb = pickle.dumps((new_info, len(fed.all_data)))
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]  # Clear data
                fed.sel.unregister(sock)  # This socket finished its job
            # Re-initialise data and socket information
            fed.all_data = {}
            fed.all_sockets = {}

            print("New model distributed!")
            print("Waiting for next communication round...")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", "\n")
            continue

        events = fed.sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                fed.accept_wrapper(key.fileobj)  # key.fileobj is the sock for the event
            else:  # Service this connection
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    try:
                        recv_header = sock.recv(1024)  # Ready to read the header.
                        msg = pickle.loads(recv_header)
                        msg_size, client_id = msg[:13].strip("*"), msg[13:].strip("*")
                        print("Preparing to receive", msg_size, "bytes from client", client_id+"...")
                        sock.setblocking(True)
                        sock.send(b"OK*"+str(client_num).encode("utf-8"))
                        print("OK")
                        recv_data = sock.recv(int(msg_size))

                        if recv_data:
                            data.outb += recv_data
                            data.id = client_id
                            # Store data and socket information for later
                            fed.all_data[client_id] = data
                            fed.all_sockets[client_id] = sock

                    except Exception as e:
                        print(e)
                        print('closing connection to', data.addr)
                        fed.sel.unregister(sock)
                        sock.close()
