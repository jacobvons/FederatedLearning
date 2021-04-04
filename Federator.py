import socket
import selectors
import types
import sys
import numpy as np
from Message import Message
from CommStage import CommStage
from Model import LinearRegression
import torch
import sklearn
import phe
from XCrypt import xcrypt_2d
from pickle import dumps, loads


class Federator:

    def __init__(self, host, port, client_num):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pk, self.sk = phe.paillier.generate_paillier_keypair()
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
        self.grads = {}
        self.biases = {}
        self.client_pks = {}
        self.pc_nums = []
        self.pcs = []

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
    # init_model = torch.load("init_model.pth")
    # init_model_message = dumps(Message(dumps(init_model), CommStage.PARAM_DIST))
    explain_ratio = 0.85
    torch.set_default_dtype(torch.float64)
    while True:
        if len(fed.conns) == fed.client_num:
            if fed.state == CommStage.CONN_ESTAB:
                print("All clients connected.")
                for sock in fed.conns:
                    # Send pk
                    sock.send(dumps(len(dumps(fed.pk))))
                    if sock.recv(2).decode("utf-8") == "OK":
                        sock.send(dumps(fed.pk))
                print("Sent public key")
                fed.conns = set()
                fed.state = CommStage.PC_INFO_EXCHANGE
            elif fed.state == CommStage.PC_INFO_EXCHANGE:
                max_pc_num = max(fed.pc_nums)
                for sock in fed.conns:
                    sock.send(dumps(max_pc_num))
                fed.conns = set()
                fed.state = CommStage.PC_AGGREGATION
            elif fed.state == CommStage.PC_AGGREGATION:
                avg_pc = sum(fed.pcs) / len(fed.pcs)  # TODO: Could implement weighted pcs
                for sock in fed.conns:
                    # encrypted_pc = xcrypt_2d(fed.client_pks[sock].encrypt, avg_pc)
                    encrypted_pc = avg_pc
                    avg_pc_msg = dumps(Message(encrypted_pc, CommStage.PC_AGGREGATION))
                    avg_pc_header = dumps(len(avg_pc_msg))
                    sock.send(avg_pc_header)
                    sock.recv(2)  # OK
                    sock.send(avg_pc_msg)
                    sock.recv(2)  # OK
                print("Sent average PC")
                # Model parameter distribution
                init_model = LinearRegression(len(avg_pc), 1)
                print("Length:", len(avg_pc))
                init_model_msg = dumps(Message(dumps(init_model), CommStage.PARAM_DIST))
                for sock in fed.conns:
                    # Send init_param, init_model
                    sock.send(dumps(len(init_model_msg)))
                    if sock.recv(2).decode("utf-8") == "OK":
                        sock.send(init_model_msg)
                fed.conns = set()
                fed.state = CommStage.REPORT
            elif fed.state == CommStage.REPORT:  # Report stage
                print("All clients reported.")
                # Aggregation and distribution
                grad_sums = 0
                bias_sums = 0
                for sock in fed.conns:
                    client_pk = fed.client_pks[sock]
                    # layer * m * n arrays
                    client_grads = np.array([xcrypt_2d(fed.sk.decrypt, g) for g in fed.grads[sock]], dtype="float64")
                    client_biases = np.array([xcrypt_2d(fed.sk.decrypt, b) for b in fed.biases[sock]], dtype="float64")
                    # client_grad = xcrypt_2d(fed.sk.decrypt, fed.grads[sock])
                    # client_bias = np.array(list(map(fed.sk.decrypt, fed.biases[sock])), dtype="float64")

                    # these two includes grads and biases of all layers
                    # Should unpack them on the client's side
                    grad_sums += client_grads
                    bias_sums += client_biases
                    # grad_sum += client_grad
                    # bias_sum += client_bias
                for sock in fed.conns:
                    client_pk = fed.client_pks[sock]
                    print(bias_sums)
                    client_grad_sums = [xcrypt_2d(client_pk.encrypt, g) for g in list(grad_sums)]
                    client_bias_sums = [xcrypt_2d(client_pk.encrypt, b) for b in list(bias_sums)]

                    grad_message = dumps(Message(client_grad_sums, CommStage.PARAM_DIST))
                    bias_message = dumps(Message(client_bias_sums, CommStage.PARAM_DIST))

                    sock.send(str(len(grad_message)).encode("utf-8"))
                    sock.recv(2)
                    sock.send(grad_message)
                    sock.send(str(len(bias_message)).encode("utf-8"))
                    sock.recv(2)
                    sock.send(bias_message)
                fed.grads = {}
                fed.biases = {}
                fed.conns = set()

        events = fed.sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                fed.accept_wrapper(key.fileobj)
            else:
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    sock.setblocking(True)
                    message = loads(sock.recv(2048))  # Of type Message
                    if message.comm_stage == CommStage.CONN_ESTAB:
                        fed.conns.add(sock)
                        fed.client_pks[sock] = message.message
                        # Send client num and explain ratio
                        sock.send(str(client_num+explain_ratio).encode("utf-8"))
                        print("Waiting to get all clients")
                    elif message.comm_stage == CommStage.PC_INFO_EXCHANGE:
                        fed.conns.add(sock)
                        fed.pc_nums.append(message.message)
                    elif message.comm_stage == CommStage.PC_AGGREGATION:
                        fed.conns.add(sock)
                        pc_header = message.message
                        sock.send(b"OK")
                        pc_msg = loads(sock.recv(pc_header))
                        # pc = xcrypt_2d(fed.sk.decrypt, pc_msg.message)
                        pc = pc_msg.message
                        fed.pcs.append(pc)
                    elif message.comm_stage == CommStage.REPORT:
                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        fed.conns.add(sock)
                        num_layers = int(message.message)
                        print(f"{num_layers} trainable layers in total")
                        for i in range(num_layers):
                            # TODO: Might have a BUG of reading too much bytes and cause blocking on client side
                            grad_header_message = loads(sock.recv(1024))
                            grad_header = grad_header_message.message
                            sock.send(b"OK")
                            grad = loads(sock.recv(int(grad_header)))  # np.array of encrypted number

                            bias_header = loads(sock.recv(1024)).message
                            sock.send(b"OK")
                            bias = loads(sock.recv(int(bias_header)))  # np.array of encrypted number
                            if sock not in fed.grads.keys():
                                fed.grads[sock] = []
                                fed.biases[sock] = []
                            fed.grads[sock].append(grad)
                            fed.biases[sock].append(bias)
                            # TODO: Clear grads and biases after aggregation
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", "\n")

                    elif message.comm_stage == CommStage.END:
                        fed.sel.unregister(sock)
                        sock.close()
                        print(f"connection to {sock} closed")
