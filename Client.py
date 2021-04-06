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
import time
import torch.nn as nn
import torch.optim as optim


class Client:

    def __init__(self, id, host, port):
        self.id = id
        self.host = host
        self.port = port
        self.fed_pk = None
        self.pk, self.sk = phe.paillier.generate_paillier_keypair()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_num = 0

    def connect(self):
        self.sock.connect((self.host, self.port))
        return True

    def send(self, message):
        self.sock.send(message)

    def recv(self, size):
        message = self.sock.recv(size)  # In bytes
        return message


if __name__ == "__main__":
    # Initialisation stage
    host, port, path, client_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    torch.set_default_dtype(torch.float64)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Connection establishment stage
    client = Client(client_id, host, port)
    client.connect()
    client.sock.setblocking(True)
    init_msg = Message(client.pk, CommStage.CONN_ESTAB)  # init msg 1
    a = dumps(init_msg)
    print("Initial message length:", len(a))
    client.send(a)  # Fed: message = loads(sock.recv(2048))
    # Receive client_num and explain_ratio
    combination = float(client.recv(10).decode("utf-8"))
    client.client_num = int(combination)
    explain_ratio = round(combination - int(combination), 2)
    print(client.client_num, "clients in total.")
    print(f"Want to explain {round(explain_ratio*100, 2)}% of data.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find number of pc for this client and report OK to federator
    data_df = pd.read_csv(path)
    features = data_df[data_df.columns[:-1]]
    features = preprocessing.normalize(features, axis=0)  # Normalise along instance axis among features
    targets = np.array(data_df[data_df.columns[-1]])
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)
    # Save training and testing sets respectively
    if not os.path.exists(f"./client{client_id}"):
        os.mkdir(f"./client{client_id}")
    np.save(f"./client{client_id}/X_train.npy", X_train)
    np.save(f"./client{client_id}/X_test.npy", X_test)
    np.save(f"./client{client_id}/y_train.npy", y_train)
    np.save(f"./client{client_id}/y_test.npy", y_test)
    print("Saved normalised original data.")
    pca = PCA(n_components=5)
    while True:
        pca.fit(X_train)
        if sum(pca.explained_variance_ratio_) >= explain_ratio:
            pc_num = len(pca.components_)
            break
        else:
            pca = PCA(n_components=pca.n_components+1)
    print("At least", pc_num, "PCs")
    client.send(b"OK")  # No.1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Conn establish (federator got all clients)
    # Receive Federator public key
    fed_pk_header = loads(client.recv(1024))  # Fed: sock.send(dumps(len(dumps(fed.pk))))
    client.send(b"OK")  # No.2
    client.fed_pk = loads(client.recv(fed_pk_header))
    print("Received Federator public key")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PC info exchange stage
    preprocess_init_msg = Message(pc_num, CommStage.PC_INFO_EXCHANGE)
    client.send(dumps(preprocess_init_msg))  # init msg 2

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PC info exchange stage (federator got all clients)
    final_pc_num = loads(client.recv(10))  # Fed: sock.send(dumps(max_pc_num))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perform PCA (Local)
    pca = PCA(n_components=final_pc_num)
    pca.fit(X_train)
    pcs = pca.components_

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PC aggregation stage
    # Send pc to Federator
    pc_msg = dumps(Message(pcs, CommStage.PC_AGGREGATION))
    pc_header_msg = Message(len(pc_msg), CommStage.PC_AGGREGATION)
    print("Sending encrypted PC")
    client.send(dumps(pc_header_msg))  # init msg 3
    client.recv(2)  # No.3

    client.send(pc_msg)
    print("Sent")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PC aggregation stage (federator got all clients)
    avg_pc_header = loads(client.recv(1024))  # Fed: sock.send(avg_pc_header)
    client.send(b"OK")  # No.4
    avg_pc_msg = loads(client.recv(avg_pc_header))
    client.send(b"OK")  # No.5

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculate and save reduced data (Local)
    avg_pc = avg_pc_msg.message
    reduced_X_train = X_train @ avg_pc.T
    reduced_X_test = X_test @ avg_pc.T
    np.save(f"./client{client_id}/reduced_X_train.npy", reduced_X_train)
    np.save(f"./client{client_id}/reduced_X_test.npy", reduced_X_test)
    print("Reduced dimensionality of original data")
    reduced_X_train = torch.from_numpy(reduced_X_train)
    reduced_X_test = torch.from_numpy(reduced_X_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Receive initial model stage
    header = client.recv(1024)  # Fed: sock.send(dumps(len(init_model_msg)))
    client.send(b"OK")  # No.6
    model_msg = client.recv(int(loads(header)))  # Fed: sock.send(init_model_msg)
    model = loads(loads(model_msg).message)
    torch.save(model, "client"+client_id+"_model.pt")
    print("Received model message")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reporting stage
    for _ in range(1):  # Number of communication rounds
        model = torch.load("client"+client_id+"_model.pt")
        # Training
        # TODO: Training
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_func = nn.MSELoss()
        model.train()
        for i in range(len(reduced_X_train)):
            optimizer.zero_grad()
            prediction = model(reduced_X_train[i])
            print(type(prediction))
            print(type(y_train[i]))
            loss = loss_func(prediction, y_train[i])
            loss.backward()
            optimizer.step()
            print(i)
        print("Done training")

        model_grads = []
        model_biases = []
        print("Encrypting model message")
        for layer in model.layers:
            grad = dumps(xcrypt_2d(client.fed_pk.encrypt, np.array(layer.weight.data, dtype="float64")))
            bias = dumps(np.array(list(map(client.fed_pk.encrypt, np.array(layer.bias.data, dtype="float64")))))
            model_grads.append(grad)
            model_biases.append(bias)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Report stage (client done training)
        print("Sending updates")
        report_start_msg = Message(len(model_grads), CommStage.REPORT)
        client.send(dumps(report_start_msg))  # init msg 4
        print(f"Sending {len(model_grads)} trainable layers")
        for i in range(len(model_grads)):
            model_grad = model_grads[i]
            model_bias = model_biases[i]

            grad_header = Message(len(model_grad), CommStage.REPORT)
            client.send(dumps(grad_header))
            client.recv(2)  # No.7
            client.send(model_grad)
            print("Sent grad")

            bias_header = Message(len(model_bias), CommStage.REPORT)
            client.send(dumps(bias_header))
            client.recv(2)  # No.8
            client.send(model_bias)
            print("Sent bias")

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Receive updated model info (federator got all grads and biases)
        grad_header = int(client.recv(10).decode("utf-8"))  # Fed: sock.send(str(len(grad_message)).encode("utf-8"))
        client.send(b"OK")  # No.9
        new_grads = loads(client.recv(grad_header)).message  # of type np.array with encrypted numbers
        bias_header = int(client.recv(10).decode("utf-8"))
        client.send(b"OK")  # No.10
        new_biases = loads(client.recv(bias_header)).message  # of type np.array with encrypted numbers

        for i in range(len(model.layers)):
            layer = model.layers[i]
            new_layer_grad = xcrypt_2d(client.sk.decrypt, new_grads[i])
            new_layer_bias = xcrypt_2d(client.sk.decrypt, new_biases[i])
            with torch.no_grad():
                layer.weight.data = torch.from_numpy(new_layer_grad)
                layer.bias.data = torch.from_numpy(new_layer_bias)

        torch.save(model, "client"+client_id+"_model.pt")
        print("New model saved.")
        print(model.linear.weight.data)
        print(model.linear.bias.data)

    # Send END message
    end_msg = Message(b"", CommStage.END)
    client.send(dumps(end_msg))
    client.sock.close()
    print("end")
