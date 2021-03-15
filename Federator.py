import socket
import selectors
import types
import sys
import pickle
import numpy as np
from Message import Message
from CommStage import CommStage
import torch
import sklearn


class Federator:

    def __init__(self, host, port, client_num):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        print('listening on', (host, port))
        self.sock.setblocking(True)
        self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        self.all_data = {}
        self.all_sockets = {}
        self.client_num = client_num
        self.conns = set()
        self.state = CommStage.CONN_ESTAB

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()
        print("Accepted connection from", addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"", id="")
        event_actions = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, event_actions, data=data)

    # def received_dummy(self):
    #     self.sock.setblocking(True)
    #     conn, addr = self.sock.accept()
    #     conn.setblocking(True)
    #     dummy = conn.recv(5).decode("utf-8")
    #     if dummy == "dummy":
    #         self.conns[addr] = conn
    #         return True
    #     return False


if __name__ == "__main__":
    host, port, client_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fed = Federator(host, port, client_num)
    init_model = torch.load("init_model.pth")
    init_model_message = pickle.dumps(Message(pickle.dumps(init_model), CommStage.PARAM_DIST))

    while True:
        if len(fed.conns) == fed.client_num:
            print("All clients connected.")
            if fed.state == CommStage.CONN_ESTAB:
                for sock in fed.conns:
                    # Send message about the init_param, init_model, pk, and client_num
                    sock.send(pickle.dumps(len(init_model_message)))
                    if sock.recv(2).decode("utf-8") == "OK":
                        sock.send(init_model_message)
                fed.conns = set()
                fed.state = CommStage.REPORT
            else:
                # TODO: Do aggregation and distribution
                fed.conns = set()
                pass


        events = fed.sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                fed.accept_wrapper(key.fileobj)
            else:
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    sock.setblocking(True)
                    message = pickle.loads(sock.recv(1024))  # Of type Message
                    if message.comm_stage == CommStage.CONN_ESTAB:
                        fed.conns.add(sock)
                        sock.send(b"OK")
                        print("Waiting to get all clients")
                    elif message.comm_stage == CommStage.REPORT:
                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        grad_header = message.message
                        print(grad_header)
                        sock.send(b"OK")
                        grad = pickle.loads(sock.recv(int(grad_header)))
                        print(grad)

                        bias_header = pickle.loads(sock.recv(1024)).message
                        print(bias_header)
                        sock.send(b"OK")
                        bias = pickle.loads(sock.recv(int(bias_header)))
                        print(bias)
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                    elif message.comm_stage == CommStage.END:
                        fed.sel.unregister(sock)
                        print(f"connection to {sock} closed")

        # # Aggregation when data from all clients are present
        # if len(fed.all_data) == fed.client_num:
        #     print("Aggregating...")
        #     # Turn bytes info of data.outb into actual arrays
        #     for data in fed.all_data.values():
        #         data.outb = pickle.loads(data.outb)
        #     # Sum them and update all
        #     new_info = sum([data.outb for data in fed.all_data.values()])
        #
        #     # Send out new parameters after aggregation
        #     print("Distributing new model...")
        #     for client_id, data in fed.all_data.items():
        #         sock = fed.all_sockets[client_id]
        #         print('Sending information to', data.addr, "(client", data.id + ")...")
        #         data.outb = pickle.dumps((new_info, len(fed.all_data)))
        #         sent = sock.send(data.outb)  # Should be ready to write
        #         data.outb = data.outb[sent:]  # Clear data
        #         fed.sel.unregister(sock)  # This socket finished its job
        #     # Re-initialise data and socket information
        #     fed.all_data = {}
        #     fed.all_sockets = {}
        #
        #     print("New model distributed!")
        #     print("Waiting for next communication round...")
        #     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        #     print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", "\n")
        #     continue
        #
        # events = fed.sel.select(timeout=None)
        # for key, mask in events:
        #     if key.data is None:
        #         fed.accept_wrapper(key.fileobj)  # key.fileobj is the socket for the event
        #     else:  # Service this connection
        #         sock = key.fileobj
        #         data = key.data
        #         if mask & selectors.EVENT_READ:
        #             try:
        #                 sock.setblocking(True)
        #                 recv_header = sock.recv(1024)  # Ready to read the header.
        #                 msg = pickle.loads(recv_header)
        #                 # Format header
        #                 msg_size, client_id = msg[:13].strip("*"), msg[13:].strip("*")
        #                 print("Preparing to receive", msg_size, "bytes from client", client_id+"...")
        #                 sock.send(b"OK*"+str(client_num).encode("utf-8"))
        #                 print("OK")
        #                 recv_data = sock.recv(int(msg_size))
        #
        #                 if recv_data:
        #                     data.outb += recv_data
        #                     data.id = client_id
        #                     # Store data and socket information for later
        #                     fed.all_data[client_id] = data
        #                     fed.all_sockets[client_id] = sock
        #
        #             except Exception as e:
        #                 print(e)
        #                 print('closing connection to', data.addr)
        #                 fed.sel.unregister(sock)
        #                 sock.close()
