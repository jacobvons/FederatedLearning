import socket
import argparse
from pickle import dumps, loads
import os
import numpy as np
import pandas as pd
from Message import Message
from CommStage import CommStage
import torch
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
from GeneralFunc import format_msg
from Crypto.PublicKey import RSA
from XCrypt import seg_encrypt, seg_decrypt


class Client:

    def __init__(self, client_id, host, port, path):
        self.client_id = client_id
        self.host = host
        self.port = port
        self.path = path
        self.fed_pk = None
        self.dir_name = None
        self.sk = RSA.generate(2048)
        self.pk = self.sk.publickey()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_num = 0
        self.current_round = 1
        self.xcrypt = True

    def connect(self):
        self.sock.connect((self.host, self.port))
        self.sock.setblocking(True)

    def send(self, message):
        self.sock.send(message)

    def recv(self, size):
        message = self.sock.recv(size)
        return message

    def send_ok(self):
        self.send(b"OK")

    def recv_ok(self):
        self.recv(2)

    def recv_large(self):
        data = b""
        while True:
            pack = self.recv(10)
            data += pack
            if data[-3:] == b"end":
                break
        return data[:-3]

    def work(self):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Connection establishment stage (single)
        pk_pem = self.pk.exportKey()
        init_msg = Message(pk_pem, CommStage.CONN_ESTAB)
        self.send(format_msg(dumps(init_msg)))  # init msg 1
        # Receive client_num and explain_ratio
        self.client_num, self.explain_ratio, self.comm_rounds, self.xcrypt, self.epoch_num, self.dir_name = loads(self.recv_large())
        print(self.client_num, "clients in total.")
        print(f"Want to explain {round(self.explain_ratio * 100, 2)}% of data.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find number of pc for this client and report OK to federator
        data_df = pd.read_csv(self.path)
        features = data_df[data_df.columns[:-1]]
        features = preprocessing.normalize(features, axis=0)  # Normalise along instance axis among features
        targets = np.array(data_df[data_df.columns[-1]])
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
        # Save training and testing sets respectively
        if not os.path.exists(f"./{self.dir_name}"):
            os.mkdir(f"./{self.dir_name}")
        if not os.path.exists(f"./{self.dir_name}/client{self.client_id}"):
            os.mkdir(f"./{self.dir_name}/client{self.client_id}")
        np.save(f"./{self.dir_name}/client{self.client_id}/X_train.npy", X_train)
        np.save(f"./{self.dir_name}/client{self.client_id}/X_test.npy", X_test)
        np.save(f"./{self.dir_name}/client{self.client_id}/y_train.npy", y_train)
        np.save(f"./{self.dir_name}/client{self.client_id}/y_test.npy", y_test)
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
        self.send_ok()  # No.1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Conn establish (batch)
        # Receive Federator public key
        self.fed_pk = RSA.import_key(loads(self.recv_large()))
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
        pcs = (pca.components_, sum(pca.explained_variance_ratio_))  # components and explain ratio sum

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC aggregation stage (single)
        # Send pc to Federator
        pc_msg = format_msg(dumps(Message(pcs, CommStage.PC_AGGREGATION)))
        print("Sending encrypted PC")
        self.send(pc_msg)  # init msg 3

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC aggregation stage (batch)
        avg_pc_msg = loads(self.recv_large())
        self.send_ok()  # No.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calculate and save reduced data (Local)
        avg_pc = avg_pc_msg.message
        reduced_X_train = X_train @ avg_pc.T
        reduced_X_test = X_test @ avg_pc.T
        np.save(f"./{self.dir_name}/client{self.client_id}/reduced_X_train.npy", reduced_X_train)
        np.save(f"./{self.dir_name}/client{self.client_id}/reduced_X_test.npy", reduced_X_test)
        reduced_X_train = torch.from_numpy(reduced_X_train)
        reduced_X_test = torch.from_numpy(reduced_X_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        train_dataset = TensorDataset(reduced_X_train, y_train)
        test_dataset = TensorDataset(reduced_X_test, y_test)
        torch.save(train_dataset, f"./{self.dir_name}/client{self.client_id}/train_dataset.pt")
        torch.save(test_dataset, f"./{self.dir_name}/client{self.client_id}/test_dataset.pt")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Receive initial model stage (single)
        model_msg = self.recv_large()
        model, optimizer, loss_func = loads(model_msg).message
        torch.save(model, f"./{self.dir_name}/client{self.client_id}/client{self.client_id}_initial_model.pt")
        print("Received model message")
        self.send_ok()  # No.6.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reporting stage
        print(f"{self.comm_rounds} communication rounds in total")
        for _ in range(self.comm_rounds):  # Communication rounds
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Training (Local)
            print(f"Round {self.current_round}")
            model.train()
            loader = DataLoader(train_dataset, shuffle=True, batch_size=10)  # TODO: Pass batch size as parameter
            for n in range(self.epoch_num):  # Training epochs
                print(f"Epoch {n+1}/{self.epoch_num}")
                for i, (X, y) in enumerate(loader):  # Mini-batches
                    optimizer.zero_grad()
                    loss = 0
                    for j in range(len(X)):  # Calculate on a mini-batch
                        prediction = model(reduced_X_train[i])
                        # loss += loss_func(prediction[0], y_train[i])
                        loss += loss_func(prediction[0], y_train[i], model)
                    loss /= len(X)  # Mean loss to do back prop
                    loss.backward()
                    optimizer.step()  # Update grad and bias for each mini-batch
            print("Done training")

            model_grads = []
            model_biases = []
            print("Encrypting model message")
            for layer in model.layers:
                grad = dumps(seg_encrypt(np.array(layer.weight.data, dtype="float64"), self.fed_pk, self.xcrypt))
                bias = dumps(seg_encrypt(np.array(layer.bias.data, dtype="float64"), self.fed_pk, self.xcrypt))

                model_grads.append(grad)
                model_biases.append(bias)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Report stage (single)
            report_start_msg = Message([len(model_grads), self.current_round], CommStage.REPORT)
            self.send(format_msg(dumps(report_start_msg)))  # init msg 4
            print(f"Sending {len(model_grads)} trainable layers")
            for i in range(len(model_grads)):
                model_grad = model_grads[i]
                self.send(format_msg(model_grad))
                self.recv_ok() # No.7.5

            for i in range(len(model_biases)):
                model_bias = model_biases[i]
                self.send(format_msg(model_bias))
                self.recv_ok()  # No.8
            # TODO: Calculate a range of metric scores to be put in metrics here
            metrics = {"cross_val": 1, "size": 1, "avg": 1}
            # TODO: Send aggregation metric here (e.g. cross validation score). Need to generate a bunch of scores first
            self.send(format_msg(dumps(metrics)))
            self.recv_ok()  # No. 8.25

            self.send_ok()  # No. 8.5

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Report stage (batch)
            # Receive updated model info
            new_grads = loads(self.recv_large()).message
            self.send_ok()  # No.9.5
            new_biases = loads(self.recv_large()).message

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Update model (Local)
            print("Updating  local model")
            for i in range(len(model.layers)):
                layer = model.layers[i]
                new_layer_grad = seg_decrypt(new_grads[i], self.sk, self.xcrypt)
                new_layer_bias = seg_decrypt(new_biases[i], self.sk, self.xcrypt, True)

                with torch.no_grad():
                    layer.weight.data = torch.from_numpy(new_layer_grad)
                    layer.bias.data = torch.from_numpy(new_layer_bias)
            torch.save(model, f"./{self.dir_name}/client{self.client_id}/client{self.client_id}_model{self.current_round}.pt")
            print("New model saved.")
            print(f"Round {self.current_round} finished")
            self.current_round += 1

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # End stage (single)
        # Send END message
        end_msg = Message(b"", CommStage.END)
        self.send(format_msg(dumps(end_msg)))  # init msg 5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # End stage (batch)
        self.recv_ok()  # No.11
        self.sock.close()
        print("The end")


if __name__ == "__main__":
    """
    Start a single client (for testing mainly)
    
    Command Line Arguments
    --h: host, compulsory
    --p: port, compulsory
    --path: relative path to training data, compulsory
    --i: client id, compulsory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--h")
    parser.add_argument("--p")
    parser.add_argument("--path")
    parser.add_argument("--i")
    args = parser.parse_args()

    host = args.h
    port = int(args.p)
    path = args.path
    client_id = args.i

    torch.set_default_dtype(torch.float64)

    client = Client(client_id, host, port, path)
    client.connect()
    client.work()
