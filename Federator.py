import socket
import sys
import numpy as np
from Message import Message
from CommStage import CommStage
from Model import LinearRegression
from Model import MLPRegression
import torch
import phe
from XCrypt import xcrypt_2d
from pickle import dumps, loads
from threading import Thread
from GeneralFunc import recv_large, format_msg


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
        self.client_num = client_num
        self.conns = set()
        self.state = CommStage.CONN_ESTAB
        self.grads = {}
        self.biases = {}
        self.client_pks = {}
        self.pc_nums = []
        self.pcs = []

    def reset(self):
        """
        Revert the Federator to initial state for next batch of connections

        :return:
        """
        self.all_data = {}
        self.all_sockets = []
        self.conns = set()
        self.state = CommStage.CONN_ESTAB
        self.grads = {}
        self.biases = {}
        self.client_pks = {}
        self.pc_nums = []
        self.pcs = []

    def reset_conns(self):
        """
        Reset connections of this batch collection round

        :return:
        """
        self.conns = set()

    # Single methods
    def single_estab_connection(self, sock, message):
        """
        Establish connection to a single client

        :param sock: the socket the connection is on
        :param message: Message with CommStage CONN_ESTAB, message body is client public key
        :return:
        """
        self.client_pks[sock] = message.message  # init msg 1
        # Send client num and explain ratio
        sock.send(str(client_num + explain_ratio).encode("utf-8"))
        sock.recv(2)  # No.1
        print("Waiting to get all clients")
        self.conns.add(sock)
        self.all_sockets.append(sock)

    def single_pc_info_exchange(self, sock, message):
        """
        Adding PC number of a single thread to the Federator

        :param sock: the socket the connection is on
        :param message: Message with CommState PC_INFO_EXCHANGE, message body is PC number of the client
        :return:
        """
        self.pc_nums.append(message.message)  # init msg 2
        self.conns.add(sock)

    def single_pc_aggregation(self, sock, message):
        """
        Receiving PC from a client

        :param sock: the socket the connection is on
        :param message: Message with CommStage PC_AGGREGATION, message body is encrypted PC
        :return:
        """
        pc = message.message  # init msg 3
        self.pcs.append(pc)
        self.conns.add(sock)

    def single_report(self, sock, message):
        """
        Receive model parameters from a client and save them

        :param sock: the socket the connection is on
        :param message: Message with CommStage REPORT, message body is number of trainable layers of client
        :return:
        """
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        num_layers = int(message.message)  # init msg 4
        print(f"{num_layers} trainable layers in total")
        for i in range(num_layers):
            grad = loads(recv_large(sock))  # np.array of encrypted number
            sock.send(b"OK")  # No.7.5
            bias = loads(recv_large(sock))  # np.array of encrypted number
            if sock not in self.grads.keys():
                self.grads[sock] = []
                self.biases[sock] = []
            self.grads[sock].append(grad)
            self.biases[sock].append(bias)
            sock.send(b"OK")  # No.8.5
        sock.recv(2)  # No.8.75
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", "\n")
        self.conns.add(sock)

    def single_end(self, sock):
        """
        Collects a client for closing connection

        :param sock: the socket the connection is on
        :return:
        """
        print(f"To close connection with {sock}")
        self.conns.add(sock)

    # Batch methods
    def batch_estab_connection(self):
        """
        Send Federator's public key to all clients

        :return:
        """
        self.reset_conns()
        print("All clients connected.")
        for sock in self.all_sockets:
            # Send pk
            sock.send(format_msg(dumps(self.pk)))
            self.start_listen_thread(sock)
        print("Sent public key")
        self.state = CommStage.PC_INFO_EXCHANGE

    def batch_pc_info_exchange(self):
        """
        Send out how many PC each client should be calculating to clients

        :return:
        """
        self.reset_conns()
        max_pc_num = max(self.pc_nums)
        for sock in self.all_sockets:
            sock.send(dumps(max_pc_num))
            self.start_listen_thread(sock)
        self.state = CommStage.PC_AGGREGATION

    def batch_pc_aggregation(self):
        """
        Aggregate PCs from clients and distribute "averaged" PC to all clients

        :return:
        """
        self.reset_conns()
        avg_pc = sum(self.pcs) / len(self.pcs)  # TODO: Could implement weighted pcs
        for sock in self.all_sockets:
            encrypted_pc = avg_pc
            avg_pc_msg = format_msg(dumps(Message(encrypted_pc, CommStage.PC_AGGREGATION)))
            sock.send(avg_pc_msg)
            sock.recv(2)  # No.5
        print("Sent average PC")

        # Model parameter distribution
        init_model = LinearRegression(len(avg_pc), 1)  # TODO: Make the model general
        # init_model = MLPRegression(len(avg_pc), 8, 1, 2)
        print("Length:", len(avg_pc))
        # TODO: Integrate optimizer and loss function into the message as well
        init_model_msg = format_msg(dumps(Message(dumps(init_model), CommStage.PARAM_DIST)))
        for sock in self.all_sockets:
            sock.send(init_model_msg)
            sock.recv(2)  # No.6.5
            self.start_listen_thread(sock)
        self.state = CommStage.REPORT

    def batch_report(self):
        """
        Calculates "averaged" model information and send the outcome to clients

        :return:
        """
        self.reset_conns()
        print("All clients reported.")
        # Aggregation and distribution
        grad_sums = []
        bias_sums = []
        for sock in self.all_sockets:
            # layer * m * n arrays
            client_grads = [xcrypt_2d(self.sk.decrypt, g) for g in self.grads[sock]]
            client_biases = [xcrypt_2d(self.sk.decrypt, b) for b in self.biases[sock]]

            if not len(grad_sums):
                grad_sums = client_grads
                bias_sums = client_biases
            else:
                grad_sums = [grad_sums[i] + client_grads[i] for i in range(0, len(client_grads))]
                bias_sums = [bias_sums[i] + client_biases[i] for i in range(0, len(bias_sums))]

        for sock in self.all_sockets:
            client_pk = self.client_pks[sock]
            client_grad_sums = [xcrypt_2d(client_pk.encrypt, g) for g in grad_sums]
            client_bias_sums = [xcrypt_2d(client_pk.encrypt, b) for b in bias_sums]

            grad_message = dumps(Message(client_grad_sums, CommStage.PARAM_DIST))
            bias_message = dumps(Message(client_bias_sums, CommStage.PARAM_DIST))

            sock.send(format_msg(grad_message))
            sock.recv(2)  # No.9.5
            sock.send(format_msg(bias_message))
            self.start_listen_thread(sock)
        self.grads = {}
        self.biases = {}
        self.state = CommStage.END

    def batch_end(self):
        """
        Closing connection to all clients and reset Federator to initial state

        :return:
        """
        self.reset_conns()
        for sock in self.all_sockets:
            sock.send(b"OK")  # No.11
            sock.close()
            print(f"Connection to {sock} closed")
        self.reset()

    # Proceeding methods
    def start_listen_thread(self, sock):
        """
        Start a thread for processing a single socket connection

        :param sock: the socket the connection is on
        :return:
        """
        listen_thread = Thread(target=self.single_proceed, args=(sock, ))
        listen_thread.start()

    def batch_proceed(self):
        """
        Process all clients based on the communication stage

        :return:
        """
        if self.state == CommStage.CONN_ESTAB:
            self.batch_estab_connection()

        elif self.state == CommStage.PC_INFO_EXCHANGE:
            self.batch_pc_info_exchange()

        elif self.state == CommStage.PC_AGGREGATION:
            self.batch_pc_aggregation()

        elif self.state == CommStage.REPORT:
            self.batch_report()

        elif self.state == CommStage.END:
            self.batch_end()

    def single_proceed(self, sock):
        """
        Process a single connection on a new thread based on the communication stage

        :param sock: the socket the connection is on
        :return:
        """
        message = loads(recv_large(sock))
        single_thread = None
        if message.comm_stage == CommStage.CONN_ESTAB:
            single_thread = Thread(target=self.single_estab_connection, args=(sock, message))

        elif message.comm_stage == CommStage.PC_INFO_EXCHANGE:
            single_thread = Thread(target=self.single_pc_info_exchange, args=(sock, message))

        elif message.comm_stage == CommStage.PC_AGGREGATION:
            single_thread = Thread(target=self.single_pc_aggregation, args=(sock, message))

        elif message.comm_stage == CommStage.REPORT:
            single_thread = Thread(target=self.single_report, args=(sock, message))

        elif message.comm_stage == CommStage.END:
            single_thread = Thread(target=self.single_end, args=(sock, ))

        single_thread.start()


if __name__ == "__main__":
    host, port, client_num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    fed = Federator(host, port, client_num)
    explain_ratio = 0.85
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
