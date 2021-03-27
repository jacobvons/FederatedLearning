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
    # Connection establishment stage
    client = Client(client_id, host, port)
    client.connect()
    client.sock.setblocking(True)
    init_msg = Message(client.pk, CommStage.CONN_ESTAB)
    client.send(dumps(init_msg))
    # save client_num and explain_ratio
    combination = float(client.recv(10).decode("utf-8"))
    client.client_num = int(combination)
    explain_ratio = round(combination - int(combination), 2)
    print(client.client_num, "clients in total.")
    print(f"Want to explain {round(explain_ratio*100, 2)}% of data.")

    # PC info exchange stage
    # Normalise feature space
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
    # Receive Federator public key
    fed_pk_header = loads(client.recv(1024))
    client.send(b"OK")
    client.fed_pk = loads(client.recv(fed_pk_header))
    print("Received Federator public key")
    preprocess_init_msg = Message(pc_num, CommStage.PC_INFO_EXCHANGE)
    client.send(dumps(preprocess_init_msg))
    final_pc_num = loads(client.recv(10))
    # Perform PCA
    pca = PCA(n_components=final_pc_num)
    pca.fit(X_train)
    # PC aggregation stage
    # Encrypt pcs
    pcs = pca.components_
    # print("Encrypting PC")
    # begin = time.time()
    # encrypted_pcs = xcrypt_2d(client.fed_pk.encrypt, pcs)
    # end = time.time()
    # print(f"Done. Spent {end-begin} seconds")
    # Send pc to Federator
    pc_msg = dumps(Message(pcs, CommStage.PC_AGGREGATION))
    pc_header_msg = Message(len(pc_msg), CommStage.PC_AGGREGATION)
    print("Sending encrypted PC")
    client.send(dumps(pc_header_msg))
    ok = client.recv(2)

    client.send(pc_msg)
    print("Sent")
    avg_pc_header = loads(client.recv(1024))
    client.send(b"OK")
    avg_pc_msg = loads(client.recv(avg_pc_header))
    client.send(b"OK")
    # avg_pc = xcrypt_2d(client.sk.decrypt, avg_pc_msg.message)
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

    # Receive initial model stage
    header = client.recv(1024)
    client.send(b"OK")
    model_msg = client.recv(int(loads(header)))
    model = loads(loads(model_msg).message)
    torch.save(model, "client"+client_id+"_model.pt")
    print("Received model message")
    print(reduced_X_train.shape)
    for _ in range(1):  # Number of communication rounds
        model = torch.load("client"+client_id+"_model.pt")
        # Training stage
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
        # model_grad = dumps(xcrypt_2d(client.fed_pk.encrypt, np.array(model.weight.data, dtype="float64")))
        # model_bias = dumps(np.array(list(map(client.fed_pk.encrypt, np.array(model.bias.data, dtype="float64")))))

        # Report stage
        print("Sending updates")
        report_start_msg = Message(len(model_grads), CommStage.REPORT)
        client.send(dumps(report_start_msg))
        print("Sending different layers")
        for i in range(len(model_grads)):
            model_grad = model_grads[i]
            model_bias = model_biases[i]

            grad_header = Message(len(model_grad), CommStage.REPORT)
            client.send(dumps(grad_header))
            ok = client.recv(2)
            if ok.decode("utf-8") == "OK":  # Federator already received the grad len
                client.send(model_grad)
                print("Sent grad")

            bias_header = Message(len(model_bias), CommStage.REPORT)
            client.send(dumps(bias_header))
            ok = client.recv(2)
            if ok.decode("utf-8") == "OK":
                client.send(model_bias)
                print("Sent bias")

        # Receive updated model info
        grad_header = int(client.recv(10).decode("utf-8"))
        client.send(b"OK")
        new_grads = loads(client.recv(grad_header)).message  # of type np.array with encrypted numbers
        bias_header = int(client.recv(10).decode("utf-8"))
        client.send(b"OK")
        new_biases = loads(client.recv(bias_header)).message  # of type np.array with encrypted numbers
        print(new_grads)
        print(new_biases)
        for i in range(len(model.layers)):
            layer = model.layers[i]
            print(new_grads[i])
            print(new_biases[i])
            new_layer_grad = xcrypt_2d(client.sk.decrypt, new_grads[i])
            new_layer_bias = xcrypt_2d(client.sk.decrypt, new_biases[i])
            print(new_layer_grad)
            print(new_layer_bias)
            with torch.no_grad():
                layer.weight.data = torch.from_numpy(new_layer_grad)
                layer.bias.data = torch.from_numpy(new_layer_bias)
        # Decryption
        # new_grads = torch.from_numpy(xcrypt_2d(client.sk.decrypt, new_grad))
        # new_biases = torch.from_numpy(np.array(list(map(client.sk.decrypt, new_bias)), dtype="float64"))
        # # Update & save model
        # model.weight = torch.nn.Parameter(new_grad / client.client_num)
        # model.bias = torch.nn.Parameter(new_bias / client.client_num)
        torch.save(model, "client"+client_id+"_model.pt")
        print("New model saved.")
        print(model.linear.weight.data)
        print(model.linear.bias.data)

    # Send END message
    end_msg = Message(b"", CommStage.END)
    client.send(dumps(end_msg))
    client.sock.close()
    print("end")
