import socket
import argparse
from Message import Message
from CommStage import CommStage
from Model import *
import torch
import torch.nn as nn
import torch.optim as optim
from pickle import dumps, loads
from threading import Thread
from GeneralFunc import *
from Crypto.PublicKey import RSA
from XCrypt import seg_decrypt, seg_encrypt
from Loss import *
import time


class Federator:

    def __init__(self, host, port, client_num, comm_rounds, explain_ratio, xcrypt, epoch_num, name, pc_agg_method=None):
        """
        Initialise a Federator instance

        :param host: str, address the Federator is deployed on
        :param port: int, port the Federator is listening to
        :param client_num: int greater or equal to 1, number of expected client connections
        :param comm_rounds: int greater than 1, number of rounds of Federator-Client communication
        :param explain_ratio: float between 0 and 1, expected explain ratio for PCA
        :param xcrypt: int 1 or 0, for doing and not doing encryption during communication
        :param epoch_num: int greater or equal to 1, number of epochs for each training round
        :param name: the name of this Federator, contains information about model, loss function, etc.
        """
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.name = name
        self.sk = RSA.generate(2048)
        self.pk = self.sk.publickey()
        while True:
            try:
                self.sock.bind((host, port))
                break
            except:
                time.sleep(1)
                continue
        self.sock.listen()
        print('listening on', (host, port))
        self.sock.setblocking(True)
        self.all_data = {}
        self.all_sockets = []
        self.client_num = client_num
        self.comm_rounds = comm_rounds
        self.explain_ratio = explain_ratio
        self.epoch_num = epoch_num
        self.conns = set()
        self.state = CommStage.CONN_ESTAB
        self.grads = {}
        self.biases = {}
        self.client_pks = {}
        self.pc_nums = []
        self.pcs = []
        self.client_ratios = []
        self.pc_agg_method = pc_agg_method  # Default average, supports by explain ratio, ...
        self.current_round = 1
        self.xcrypt = xcrypt
        self.threads = []
        self.terminate = False

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
        self.client_ratios = []
        self.current_round = 1
        self.threads = []

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
        self.client_pks[sock] = RSA.import_key(message.message)  # init msg 1
        # Send client num and explain ratio
        sock.send(
            format_msg(
                dumps(
                    [self.client_num, self.explain_ratio, self.comm_rounds, self.xcrypt, self.epoch_num, self.name]
                )
            )
        )
        recv_ok(sock)  # No.1
        print("Waiting to get all clients")
        self.conns.add(sock)
        self.all_sockets.append(sock)
        self.threads = []

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
        pc, ratio = message.message  # init msg 3
        self.pcs.append(pc)
        self.client_ratios.append(ratio)
        self.conns.add(sock)

    def single_report(self, sock, message):
        """
        Receive model parameters from a client and save them

        :param sock: the socket the connection is on
        :param message: Message with CommStage REPORT, message body is number of trainable layers of client
        :return:
        """
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        num_layers, current_round = message.message  # init msg 4
        print(f"{num_layers} trainable layers in total")
        print(f"communication round {current_round}")
        self.current_round = current_round

        if sock not in self.grads.keys():
            self.grads[sock] = []
            self.biases[sock] = []

        for i in range(num_layers):
            grad = loads(recv_large(sock))
            self.grads[sock].append(grad)
            send_ok(sock)  # No.7.5

        for i in range(num_layers):
            bias = loads(recv_large(sock))
            self.biases[sock].append(bias)
            send_ok(sock)  # No.8.5

        recv_ok(sock)  # No.8.75
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", "\n")
        self.conns.add(sock)

    def single_end(self, sock):
        """
        Collects a client for closing connection

        :param sock: the socket the connection is on
        :return:
        """
        print(f"To close connection with {sock}")
        self.state = CommStage.END
        self.conns.add(sock)

    # Batch methods
    def batch_estab_connection(self):
        """
        Send Federator's public key to all clients

        :return:
        """
        self.reset_conns()
        print("All clients connected.")
        pk_pem = self.pk.exportKey()
        for sock in self.all_sockets:
            # Send pk
            sock.send(format_msg(dumps(pk_pem)))
            self.single_proceed(sock)
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
            self.single_proceed(sock)
        self.state = CommStage.PC_AGGREGATION

    def batch_pc_aggregation(self):
        """
        Aggregate PCs from clients and distribute "averaged" PC to all clients

        :return:
        """
        self.reset_conns()
        # Weighted PC aggregation methods
        if not self.pc_agg_method or self.pc_agg_method == "avg":
            print("using avg")
            avg_pc = sum(self.pcs) / len(self.pcs)  # TODO: Implement weighted pcs
        elif self.pc_agg_method == "exp_ratio":
            print("using explain ratio")
            avg_pc = sum([self.pcs[i] * self.client_ratios[i] for i in range(len(self.pcs))]) / sum(self.client_ratios)
        else:  # If typo or any other situations, use average
            print("using default avg due to typo")
            avg_pc = sum(self.pcs) / len(self.pcs)

        for sock in self.all_sockets:
            encrypted_pc = avg_pc
            avg_pc_msg = format_msg(dumps(Message(encrypted_pc, CommStage.PC_AGGREGATION)))
            sock.send(avg_pc_msg)
            recv_ok(sock)  # No.5
        print("Sent average PC")

        # Model parameter distribution
        # init_model = LinearRegression(len(avg_pc), 1)
        init_model = MLPRegression(len(avg_pc), 8, 1, 2)
        optimizer = optim.SGD(init_model.parameters(), lr=0.01)  # TODO: Tune hyper-parameters
        # loss_func = MSELoss()
        # loss_func = RidgeLoss()
        loss_func = LassoLoss(alpha=0.001)
        print("Average PC Length:", len(avg_pc))
        init_model_msg = format_msg(dumps(Message([init_model, optimizer, loss_func], CommStage.PARAM_DIST)))
        for sock in self.all_sockets:
            sock.send(init_model_msg)
            recv_ok(sock)  # No.6.5
            self.single_proceed(sock)
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
            client_grads = [seg_decrypt(g, self.sk, self.xcrypt) for g in self.grads[sock]]
            client_biases = [seg_decrypt(b, self.sk, self.xcrypt, True) for b in self.biases[sock]]

            if not len(grad_sums):
                grad_sums = client_grads
                bias_sums = client_biases
            else:
                grad_sums = [grad_sums[i] + client_grads[i] for i in range(0, len(client_grads))]
                bias_sums = [bias_sums[i] + client_biases[i] for i in range(0, len(bias_sums))]

        for sock in self.all_sockets:
            client_pk = self.client_pks[sock]
            client_grad_sums = [seg_encrypt(g, client_pk, self.xcrypt) for g in grad_sums]
            client_bias_sums = [seg_encrypt(b, client_pk, self.xcrypt) for b in bias_sums]

            grad_message = dumps(Message(client_grad_sums, CommStage.PARAM_DIST))
            bias_message = dumps(Message(client_bias_sums, CommStage.PARAM_DIST))

            sock.send(format_msg(grad_message))
            recv_ok(sock)  # No.9.5
            sock.send(format_msg(bias_message))
        self.grads = {}
        self.biases = {}

        for sock in self.all_sockets:
            self.single_proceed(sock)

    def batch_end(self):
        """
        Closing connection to all clients and reset Federator to initial state

        :return:
        """
        self.reset_conns()
        for sock in self.all_sockets:
            send_ok(sock)  # No.11
            sock.shutdown(2)
            sock.close()
            print(f"Connection to {sock} closed")
        self.reset()
        self.terminate = True

    # Proceeding methods
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
        self.threads = []

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
        self.threads.append(single_thread)


if __name__ == "__main__":
    """
    Start the Federator
    
    Command Line Arguments
    --h: host, compulsory
    --p: port, compulsory
    --n: client number, compulsory
    --rounds: communication rounds, default 1
    --ratio: explain ratio, must be float, default 0.85
    --x: xcrypt or not, 1 or 0, default 1
    --e: number of epochs for each communication round, default 1 epoch
    --name: the name of this experiment
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--h")
    parser.add_argument("--p")
    parser.add_argument("--n")
    parser.add_argument("--rounds")
    parser.add_argument("--ratio")
    parser.add_argument("--x")
    parser.add_argument("--e")
    parser.add_argument("--name")
    args = parser.parse_args()

    host = args.h
    port = int(args.p)
    client_num = int(args.n)
    training_rounds = int(args.rounds) if args.rounds else 1
    explain_ratio = min(1.0, float(args.ratio)) if args.ratio else 0.85
    xcrypt = bool(int(args.x)) if args.x else True
    epoch_num = int(args.e) if args.e else 1
    name = args.name  # This can contain information about the models, loss functions, etc.
    fed = Federator(host, port, client_num, training_rounds, explain_ratio, xcrypt, epoch_num, name)
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
