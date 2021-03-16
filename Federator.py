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
import phe


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

    def accept_wrapper(self, sock):
        conn, addr = sock.accept()
        print("Accepted connection from", addr)
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"", id="")
        event_actions = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, event_actions, data=data)

    def xcrypt_2d(self, xcrypt, message):
        m, n = message.shape
        output = []
        for i in range(0, m):
            output.append(np.array(list(map(xcrypt, message[i]))))
        return np.array(output)


if __name__ == "__main__":
    host, port, client_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fed = Federator(host, port, client_num)
    init_model = torch.load("init_model.pth")
    init_model_message = pickle.dumps(Message(pickle.dumps(init_model), CommStage.PARAM_DIST))

    while True:
        if len(fed.conns) == fed.client_num:
            if fed.state == CommStage.CONN_ESTAB:
                print("All clients connected.")
                for sock in fed.conns:
                    # Send init_param, init_model
                    sock.send(pickle.dumps(len(init_model_message)))
                    if sock.recv(2).decode("utf-8") == "OK":
                        sock.send(init_model_message)
                    # Send pk
                    sock.send(pickle.dumps(len(pickle.dumps(fed.pk))))
                    if sock.recv(2).decode("utf-8") == "OK":
                        sock.send(pickle.dumps(fed.pk))
                fed.conns = set()
                fed.state = CommStage.REPORT
            else:
                print("All clients reported.")
                # Aggregation and distribution
                grad_sum = 0
                bias_sum = 0
                for sock in fed.conns:
                    # TODO: Do decryption and encryption here
                    client_pk = fed.client_pks[sock]
                    client_grad = fed.xcrypt_2d(fed.sk.decrypt, fed.grads[sock])
                    client_bias = np.array(list(map(fed.sk.decrypt, fed.biases[sock])), dtype="float64")

                    grad_sum += client_grad
                    bias_sum += client_bias

                for sock in fed.conns:
                    print(sock)
                    client_pk = fed.client_pks[sock]
                    client_grad_sum = fed.xcrypt_2d(client_pk.encrypt, grad_sum)
                    client_bias_sum = np.array(list(map(client_pk.encrypt, bias_sum)))

                    grad_message = pickle.dumps(Message(client_grad_sum, CommStage.PARAM_DIST))
                    bias_message = pickle.dumps(Message(client_bias_sum, CommStage.PARAM_DIST))

                    sock.send(str(len(grad_message)).encode("utf-8"))
                    sock.recv(2)
                    sock.send(grad_message)
                    sock.send(str(len(bias_message)).encode("utf-8"))
                    sock.recv(2)
                    sock.send(bias_message)
                fed.conns = set()
            # continue

        events = fed.sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                fed.accept_wrapper(key.fileobj)
            else:
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    sock.setblocking(True)
                    message = pickle.loads(sock.recv(2048))  # Of type Message
                    if message.comm_stage == CommStage.CONN_ESTAB:
                        fed.conns.add(sock)
                        fed.client_pks[sock] = message.message
                        # Send client num
                        sock.send(str(client_num).encode("utf-8"))
                        print("Waiting to get all clients")
                    elif message.comm_stage == CommStage.REPORT:
                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        fed.conns.add(sock)
                        grad_header = message.message
                        sock.send(b"OK")
                        grad = pickle.loads(sock.recv(int(grad_header)))  # np.array of encrypted number

                        bias_header = pickle.loads(sock.recv(1024)).message
                        sock.send(b"OK")
                        bias = pickle.loads(sock.recv(int(bias_header)))  # np.array of encrypted number

                        fed.grads[sock] = grad
                        fed.biases[sock] = bias
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

                    elif message.comm_stage == CommStage.END:
                        fed.sel.unregister(sock)
                        sock.close()
                        print(f"connection to {sock} closed")
