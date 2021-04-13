import socket
from pickle import dumps, loads
import sys
import os
import numpy as np
import pandas as pd
import phe
from Message import Message
from CommStage import CommStage
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from XCrypt import xcrypt_2d
import torch.nn as nn
import torch.optim as optim
from GeneralFunc import format_msg


class Client:

    def __init__(self, client_id, host, port):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.fed_pk = None
        self.pk, self.sk = phe.paillier.generate_paillier_keypair()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_num = 0
        self.current_round = 1

    def connect(self):
        self.sock.connect((self.host, self.port))
        self.sock.setblocking(True)

    def send(self, message):
        self.sock.send(message)

    def recv(self, size):
        message = self.sock.recv(size)
        return message

    def recv_large(self):
        data = b""
        while True:
            pack = self.recv(1024)
            data += pack
            if data[-3:] == b"end":
                break
        return data[:-3]

    def work(self, path):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Connection establishment stage (single)
        init_msg = Message(self.pk, CommStage.CONN_ESTAB)  # init msg 1
        self.send(format_msg(dumps(init_msg)))  # Fed: message = loads(sock.recv(2048))
        # Receive client_num and explain_ratio
        self.client_num, self.explain_ratio, self.comm_rounds = loads(self.recv_large())
        print(self.client_num, "clients in total.")
        print(f"Want to explain {round(self.explain_ratio * 100, 2)}% of data.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find number of pc for this client and report OK to federator
        data_df = pd.read_csv(path)
        features = data_df[data_df.columns[:-1]]
        features = preprocessing.normalize(features, axis=0)  # Normalise along instance axis among features
        targets = np.array(data_df[data_df.columns[-1]])
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
        # Save training and testing sets respectively
        if not os.path.exists(f"./client{self.client_id}"):
            os.mkdir(f"./client{self.client_id}")
        np.save(f"./client{self.client_id}/X_train.npy", X_train)
        np.save(f"./client{self.client_id}/X_test.npy", X_test)
        np.save(f"./client{self.client_id}/y_train.npy", y_train)
        np.save(f"./client{self.client_id}/y_test.npy", y_test)
        print("Saved normalised original data.")
        pca = PCA(n_components=5)
        while True:
            pca.fit(X_train)
            if sum(pca.explained_variance_ratio_) >= self.explain_ratio:
                pc_num = len(pca.components_)
                break
            else:
                pca = PCA(n_components=pca.n_components + 1)
        print("At least", pc_num, "PCs")
        self.send(b"OK")  # No.1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Conn establish (batch)
        # Receive Federator public key
        self.fed_pk = loads(self.recv_large())
        print("Received Federator public key")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC info exchange stage (single)
        preprocess_init_msg = Message(pc_num, CommStage.PC_INFO_EXCHANGE)
        self.send(format_msg(dumps(preprocess_init_msg)))  # init msg 2

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC info exchange stage (batch)
        final_pc_num = loads(self.recv(10))  # Fed: sock.send(dumps(max_pc_num))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform PCA (Local)
        pca = PCA(n_components=final_pc_num)
        pca.fit(X_train)
        pcs = pca.components_

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC aggregation stage (single)
        # Send pc to Federator
        pc_msg = format_msg(dumps(Message(pcs, CommStage.PC_AGGREGATION)))
        print("Sending encrypted PC")
        self.send(pc_msg)  # init msg 3
        print("Sent")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC aggregation stage (batch)
        avg_pc_msg = loads(self.recv_large())
        self.send(b"OK")  # No.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate and save reduced data (Local)
        avg_pc = avg_pc_msg.message
        reduced_X_train = X_train @ avg_pc.T
        reduced_X_test = X_test @ avg_pc.T
        np.save(f"./client{self.client_id}/reduced_X_train.npy", reduced_X_train)
        np.save(f"./client{self.client_id}/reduced_X_test.npy", reduced_X_test)
        print("Reduced dimensionality of original data")
        reduced_X_train = torch.from_numpy(reduced_X_train)
        y_train = torch.from_numpy(y_train)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Receive initial model stage (single)
        model_msg = self.recv_large()
        model = loads(loads(model_msg).message)
        torch.save(model, f"./client{self.client_id}/client{self.client_id}_initial_model.pt")
        print("Received model message")
        self.send(b"OK")  # No.6.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reporting stage
        for _ in range(self.comm_rounds):  # TODO: Number of communication rounds
            model = torch.load(f"./client{self.client_id}/client{self.client_id}_initial_model.pt")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training stage (Local)
            # TODO: Receive optimizer and loss function from Federator as well
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            loss_func = nn.MSELoss()
            model.train()
            for i in range(len(reduced_X_train)):
                optimizer.zero_grad()
                prediction = model(reduced_X_train[i])

                loss = loss_func(prediction, y_train[i])
                loss.backward()
                optimizer.step()
            print("Done training")

            model_grads = []
            model_biases = []
            print("Encrypting model message")
            for layer in model.layers:
                grad = dumps(xcrypt_2d(self.fed_pk.encrypt, np.array(layer.weight.data, dtype="float64")))
                bias = dumps(np.array(list(map(self.fed_pk.encrypt, np.array(layer.bias.data, dtype="float64")))))
                model_grads.append(grad)
                model_biases.append(bias)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Report stage (single)
            print("Sending updates")
            report_start_msg = Message([len(model_grads), self.current_round], CommStage.REPORT)
            self.send(format_msg(dumps(report_start_msg)))  # init msg 4
            print(f"Sending {len(model_grads)} trainable layers")
            for i in range(len(model_grads)):
                model_grad = model_grads[i]
                model_bias = model_biases[i]

                self.send(format_msg(model_grad))
                print("Sent grad")
                self.recv(2)  # No.7.5
                self.send(format_msg(model_bias))
                print("Sent bias")
                self.recv(2)  # No.8.5
            self.send(b"OK")  # No.8.75

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Report stage (batch)
            # Receive updated model info
            new_grads = loads(self.recv_large()).message
            self.send(b"OK")  # No.9.5
            new_biases = loads(self.recv_large()).message

            for i in range(len(model.layers)):
                layer = model.layers[i]
                new_layer_grad = xcrypt_2d(self.sk.decrypt, new_grads[i])
                new_layer_bias = xcrypt_2d(self.sk.decrypt, new_biases[i])
                with torch.no_grad():
                    layer.weight.data = torch.from_numpy(new_layer_grad)
                    layer.bias.data = torch.from_numpy(new_layer_bias)

            torch.save(model, f"./client{self.client_id}/client{self.client_id}_model.pt")
            print("New model saved.")
            self.current_round += 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # End stage (single)
        # Send END message
        end_msg = Message(b"", CommStage.END)
        self.send(format_msg(dumps(end_msg)))  # init msg 5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # End stage (batch)
        self.recv(2)  # No.11
        self.sock.close()
        print("The end")


if __name__ == "__main__":
    # Used for creating single client (mainly for testing) 
    host, port, path, client_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    torch.set_default_dtype(torch.float64)

    client = Client(client_id, host, port)
    client.connect()
    client.work(path)
