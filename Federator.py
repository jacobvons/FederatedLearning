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
from threading import Thread


class Federator:

    def __init__(self, host, port, client_num):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.pk, self.sk = phe.paillier.generate_paillier_keypair()
        self.sock.bind((host, port))
        self.sock.listen()
        print('listening on', (host, port))
        self.sock.setblocking(True)
        self.all_data = {}
        self.all_sockets = []
        self.listen_threads = []
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

    def estab_connection(self, sock, message):
        self.client_pks[sock] = message.message  # init msg 1
        # Send client num and explain ratio
        sock.send(str(client_num + explain_ratio).encode("utf-8"))
        sock.recv(2)  # No.1
        print("Waiting to get all clients")
        self.conns.add(sock)
        print(len(self.conns))
        self.all_sockets.append(sock)

    def pc_info_exchange(self, sock, message):
        self.pc_nums.append(message.message)  # init msg 2
        self.conns.add(sock)

    def pc_aggregation(self, sock, message):
        pc_header = message.message  # init msg 3
        sock.send(b"OK")  # No.3
        pc_msg = loads(sock.recv(pc_header))
        pc = pc_msg.message
        self.pcs.append(pc)
        self.conns.add(sock)

    def report(self, sock, message):
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        num_layers = int(message.message)  # init msg 4
        print(f"{num_layers} trainable layers in total")
        for i in range(num_layers):
            # TODO: Might have a BUG of reading too much bytes and cause blocking on client side
            grad_header_message = loads(sock.recv(1024))
            grad_header = grad_header_message.message
            sock.send(b"OK")  # No.7
            grad = loads(sock.recv(int(grad_header)))  # np.array of encrypted number

            bias_header = loads(sock.recv(1024)).message
            sock.send(b"OK")  # No.8
            bias = loads(sock.recv(int(bias_header)))  # np.array of encrypted number

            if sock not in self.grads.keys():
                self.grads[sock] = []
                self.biases[sock] = []
            self.grads[sock].append(grad)
            self.biases[sock].append(bias)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", "\n")
        self.conns.add(sock)

    def end(self, sock):
        sock.close()
        print(f"connection to {sock} closed")

    def start_listen_thread(self, sock):
        listen_thread = Thread(target=self.single_proceed, args=(sock,))
        self.listen_threads.append(listen_thread)
        listen_thread.start()

    def batch_proceed(self):

        if self.state == CommStage.CONN_ESTAB:
            self.conns = set()
            print("All clients connected.")
            for sock in self.all_sockets:
                # Send pk
                sock.send(dumps(len(dumps(self.pk))))
                sock.recv(2)  # No.2
                sock.send(dumps(self.pk))
                self.start_listen_thread(sock)

            print("Sent public key")
            self.state = CommStage.PC_INFO_EXCHANGE

        elif self.state == CommStage.PC_INFO_EXCHANGE:
            self.conns = set()
            max_pc_num = max(self.pc_nums)
            print("HERE")
            for sock in self.all_sockets:
                sock.send(dumps(max_pc_num))
                self.start_listen_thread(sock)
            self.state = CommStage.PC_AGGREGATION

        elif self.state == CommStage.PC_AGGREGATION:
            self.conns = set()
            avg_pc = sum(self.pcs) / len(self.pcs)  # TODO: Could implement weighted pcs
            for sock in self.all_sockets:
                encrypted_pc = avg_pc
                avg_pc_msg = dumps(Message(encrypted_pc, CommStage.PC_AGGREGATION))
                avg_pc_header = dumps(len(avg_pc_msg))
                sock.send(avg_pc_header)
                sock.recv(2)  # No.4
                sock.send(avg_pc_msg)
                sock.recv(2)  # No.5
            print("Sent average PC")

            # Model parameter distribution
            init_model = LinearRegression(len(avg_pc), 1)  # TODO: Make the model general
            print("Length:", len(avg_pc))
            init_model_msg = dumps(Message(dumps(init_model), CommStage.PARAM_DIST))
            for sock in self.all_sockets:
                # Send init_param, init_model
                sock.send(dumps(len(init_model_msg)))
                sock.recv(2)  # No.6
                sock.send(init_model_msg)
                self.start_listen_thread(sock)
                self.state = CommStage.REPORT

        elif self.state == CommStage.REPORT:  # Report stage
            self.conns = set()
            print("All clients reported.")
            # Aggregation and distribution
            grad_sums = 0
            bias_sums = 0
            for sock in self.all_sockets:
                # layer * m * n arrays
                client_grads = np.array([xcrypt_2d(self.sk.decrypt, g) for g in self.grads[sock]], dtype="float64")
                client_biases = np.array([xcrypt_2d(self.sk.decrypt, b) for b in self.biases[sock]], dtype="float64")

                # these two includes grads and biases of all layers
                # Should unpack them on the client's side
                grad_sums += client_grads
                bias_sums += client_biases

            for sock in self.all_sockets:
                client_pk = self.client_pks[sock]
                client_grad_sums = [xcrypt_2d(client_pk.encrypt, g) for g in list(grad_sums)]
                client_bias_sums = [xcrypt_2d(client_pk.encrypt, b) for b in list(bias_sums)]

                grad_message = dumps(Message(client_grad_sums, CommStage.PARAM_DIST))
                bias_message = dumps(Message(client_bias_sums, CommStage.PARAM_DIST))

                sock.send(str(len(grad_message)).encode("utf-8"))
                sock.recv(2)  # No.9
                sock.send(grad_message)
                sock.send(str(len(bias_message)).encode("utf-8"))
                sock.recv(2)  # No.10
                sock.send(bias_message)
                self.start_listen_thread(sock)
            self.grads = {}
            self.biases = {}

    def single_proceed(self, sock):
        message = loads(sock.recv(2048))  # initial messages

        if message.comm_stage == CommStage.CONN_ESTAB:
            thread = Thread(target=self.estab_connection, args=(sock, message))
            thread.start()

        elif message.comm_stage == CommStage.PC_INFO_EXCHANGE:
            thread = Thread(target=self.pc_info_exchange, args=(sock, message))
            thread.start()

        elif message.comm_stage == CommStage.PC_AGGREGATION:
            thread = Thread(target=self.pc_aggregation, args=(sock, message))
            thread.start()

        elif message.comm_stage == CommStage.REPORT:
            thread = Thread(target=self.report, args=(sock, message))
            thread.start()

        elif message.comm_stage == CommStage.END:
            thread = Thread(target=self.end, args=(sock,))
            thread.start()


if __name__ == "__main__":
    host, port, client_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fed = Federator(host, port, client_num)
    explain_ratio = 0.85
    torch.set_default_dtype(torch.float64)

    while True:

        if len(fed.all_sockets) < fed.client_num:  # Collecting client connections
            fed.sock.settimeout(0.1)  # 3s timeout
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
            thread.join()  # Do NOT accept new connections until this process is done

        fed.listen_threads = [t for t in fed.listen_threads if t.is_alive()]
