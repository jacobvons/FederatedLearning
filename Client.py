import socket
import argparse
from pickle import dumps, loads
import os
import numpy as np
import pandas as pd
from Message import Message
from CommStage import CommStage
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from GeneralFunc import format_msg
from Crypto.PublicKey import RSA
from XCrypt import seg_encrypt, seg_decrypt
from Loss import *


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
        self.metrics = {"avg": 1}  # Initialising average as an aggregation metric

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_num = 0
        self.current_round = 1
        self.xcrypt = True
        self.lr = 1

        if not os.path.exists("./tests"):
            os.mkdir("./tests")
        if not os.path.exists(f"./tests/{self.dir_name}"):
            os.mkdir(f"./tests/{self.dir_name}")
        if not os.path.exists(f"./tests/{self.dir_name}/client{self.client_id}"):
            os.mkdir(f"./tests/{self.dir_name}/client{self.client_id}")
        self.client_dir = f"./tests/{self.dir_name}/client{self.client_id}"

    def connect(self):
        """
        Establishes connection to the host
        :return:
        """
        self.sock.connect((self.host, self.port))
        self.sock.setblocking(True)

    def send(self, message):
        """
        Send a message through client socket
        :param message: Message got sent
        :return:
        """
        self.sock.send(message)

    def recv(self, size):
        """
        Receives a bytes from host
        :param size: Expected byte size
        :return: bytes of size "size"
        """
        message = self.sock.recv(size)
        return message

    def send_ok(self):
        """
        Sends OK for confirmation
        :return:
        """
        self.send(b"OK")

    def recv_ok(self):
        """
        Receives OK from host for confirmation
        :return:
        """
        self.recv(2)

    def recv_large(self):
        """
        Receive a "large" piece of Message iteratively until it finishes
        :return: the ACTUAL message without b'end'
        """
        data = b""
        while True:
            pack = self.recv(10)
            data += pack
            if data[-3:] == b"end":
                break
        return data[:-3]

    def local_train(self, model, optimizer, loss_func, train_dataset):
        """
        Perform training on local data
        :param model: model
        :param optimizer: optimizer
        :param loss_func: loss function
        :param train_dataset: local training dataset
        :return: model and optimizer after training
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Training (Local)
        # Update learning rate (constant, increasing or descending)
        for g in optimizer.param_groups:
            g["lr"] = g["lr"] * self.lr
        kfold = KFold(n_splits=5, shuffle=True)  # TODO: Pass n_splits as a parameter
        model.train()
        # K-fold cross validation
        for fold, (train_inds, val_inds) in enumerate(kfold.split(train_dataset)):
            self.metrics["cross_val"] = 0
            train_sampler = SubsetRandomSampler(train_inds)
            val_sampler = SubsetRandomSampler(val_inds)

            train_loader = DataLoader(train_dataset, batch_size=10, sampler=train_sampler)
            val_loader = DataLoader(train_dataset, batch_size=10, sampler=val_sampler)
            # Training epochs
            for n in range(self.epoch_num):  # Training epochs
                print(f"Epoch {n + 1}/{self.epoch_num}")
                # Mini batches
                for i, (X, y) in enumerate(train_loader, 0):  # Mini-batches
                    optimizer.zero_grad()  # Reset optimizer parameters
                    loss = 0
                    # A mini batch
                    for j in range(len(X)):  # Calculate on a mini-batch
                        prediction = model(X[j])
                        loss += loss_func(prediction[0], y[j], model)
                    loss /= len(X)  # Mean loss to do back prop
                    loss.backward()
                    optimizer.step()  # Update grad and bias for each mini-batch

            # Cross validation using validation set
            with torch.no_grad():
                cv_loss_func = MSELoss()  # Use a separate loss function for cross validation
                cv_loss = 0
                for j, (features, target) in enumerate(val_loader, 0):
                    prediction = model(features)
                    cv_loss += float(cv_loss_func(prediction[0], target, model))
            self.metrics["cross_val"] += cv_loss  # Adding cv as an aggregation metric
        self.metrics["cross_val"] = self.epoch_num / self.metrics["cross_val"]  # cv_score = 1 / (mse / epoch)
        print("Done training")
        return model, optimizer

    def train_and_report(self, model, optimizer, loss_func, train_dataset):
        """
        Performs local training with given model and some compulsory communication with host
        :param model: model used for training
        :param optimizer: optimizer
        :param loss_func: loss function
        :param train_dataset: PyTorch Dataset instance, used for training, containing the validation data
        :return:
        """
        for _ in range(self.comm_rounds):  # Communication rounds
            print(f"Round {self.current_round}")
            model, optimizer = self.local_train(model, optimizer, loss_func, train_dataset)

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
                self.recv_ok()  # No.7.5

            for i in range(len(model_biases)):
                model_bias = model_biases[i]
                self.send(format_msg(model_bias))
                self.recv_ok()  # No.8

            # Sending metric scores
            self.send(format_msg(dumps(self.metrics)))
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
            torch.save(model, os.path.join(self.client_dir, f"client{self.client_id}_model{self.current_round}.pt"))
            print("New model saved.")
            print(f"Round {self.current_round} finished")
            self.current_round += 1

    def find_pc_num(self, X_train):
        """
        Find number of principle components of a given trainig set
        :param X_train: training data to be reduced
        :return: number of principle components expected
        """
        pca = PCA(n_components=5)
        while True:
            pca.fit(X_train)
            if sum(pca.explained_variance_ratio_) >= self.explain_ratio:
                pc_num = len(pca.components_)
                break
            else:
                pca = PCA(n_components=pca.n_components + 1)
        print("At least", pc_num, "PCs")
        return pc_num

    def save_original_data(self, X_train, X_test, y_train, y_test):
        """
        Saves the given data to numpy files
        :param X_train: training feature data
        :param X_test: testing feature data
        :param y_train: training target data
        :param y_test: testing target data
        :return:
        """
        # Save training and testing sets respectively
        np.save(os.path.join(self.client_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.client_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.client_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.client_dir, "y_test.npy"), y_test)
        print("Saved normalised original data.")

    def save_reduced_dataset(self, avg_pc, X_train, X_test, y_train, y_test):
        """
        Reduce training data, generate TensorDataset and save them
        :param avg_pc: "averaged" principle components used for dimension reduction
        :param X_train: training feature data
        :param X_test: testing feature data
        :param y_train: training target data
        :param y_test: testing target data
        :return: training dataset and testing dataset
        """
        reduced_X_train = torch.from_numpy(X_train @ avg_pc.T)
        reduced_X_test = torch.from_numpy(X_test @ avg_pc.T)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        train_dataset = TensorDataset(reduced_X_train, y_train)
        test_dataset = TensorDataset(reduced_X_test, y_test)

        torch.save(train_dataset, os.path.join(self.client_dir, "train_dataset.pt"))
        torch.save(test_dataset, os.path.join(self.client_dir, "test_dataset.pt"))
        return train_dataset, test_dataset

    def dim_reduction(self, X_train):
        """
        Perform dimension reduction and do some compulsory communication with host
        :param X_train: training feature data
        :return: pcs: principle components and the final explain ratio
        """
        pc_num = self.find_pc_num(X_train)
        self.send_ok()  # No.1
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Conn establish (batch)
        # Receive Federator public key
        self.fed_pk = RSA.import_key(loads(self.recv_large()))
        print("Received Federator public key")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC info exchange stage (single)
        preprocess_init_msg = Message([pc_num, X_train.shape[0]], CommStage.PC_INFO_EXCHANGE)
        self.send(format_msg(dumps(preprocess_init_msg)))  # init msg 2

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PC info exchange stage (batch)
        final_pc_num = loads(self.recv(10))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Perform PCA (Local)
        pca = PCA(n_components=final_pc_num)
        pca.fit(X_train)
        pcs = (pca.components_, sum(pca.explained_variance_ratio_))  # principle components and explain ratio sum
        return pcs

    def work(self):
        """
        Client working pipeline
        :return:
        """
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Connection establishment stage (single)
        pk_pem = self.pk.exportKey()
        init_msg = Message(pk_pem, CommStage.CONN_ESTAB)
        self.send(format_msg(dumps(init_msg)))  # init msg 1
        # Receive client_num and explain_ratio
        self.client_num, self.explain_ratio, self.comm_rounds, self.xcrypt, self.epoch_num, self.dir_name, self.lr = loads(self.recv_large())
        print(self.client_num, "clients in total.")
        print(f"Want to explain {round(self.explain_ratio * 100, 2)}% of data.")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Find number of pc for this client and report OK to federator
        data_df = pd.read_csv(self.path)
        features = data_df[data_df.columns[:-1]]
        features = preprocessing.normalize(features, axis=0)  # Normalise along instance axis among features
        targets = np.array(data_df[data_df.columns[-1]])
        # Adding size as an aggregation metric
        self.metrics["size"] = targets.shape[0]
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
        self.save_original_data(X_train, X_test, y_train, y_test)
        pcs = self.dim_reduction(X_train)

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
        train_dataset, test_dataset = self.save_reduced_dataset(avg_pc, X_train, X_test, y_train, y_test)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Receive initial model stage (single)
        model_msg = self.recv_large()
        model, optimizer, loss_func = loads(model_msg).message
        torch.save(model, os.path.join(self.client_dir, f"client{self.client_id}_initial_model.pt"))
        print("Received model message")
        self.send_ok()  # No.6.5

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Reporting stage
        print(f"{self.comm_rounds} communication rounds in total")
        # Local Training
        self.train_and_report(model, optimizer, loss_func, train_dataset)

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
