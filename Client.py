import socket
import pickle
import sys
import numpy as np
import phe
from Message import Message
from CommStage import CommStage
import torch
import sklearn


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

    def send_dummy(self):
        self.sock.send(b"dummy")
        return True

    def wait_conn(self):
        self.sock.setblocking(True)
        status = self.sock.recv(9).decode("utf-8")  # Should receive a "CONNECTED" message from client
        return status == "CONNECTED"

    def xcrypt_2d(self, xcrypt, message):
        m, n = message.shape
        output = []
        for i in range(0, m):
            output.append(np.array(list(map(xcrypt, message[i]))))
        return np.array(output)

    def dumps(self, message):
        output = pickle.dumps(message)
        return output

    def loads(self, message):
        output = pickle.loads(message)
        return output

    # Generate header, 0-12 is right * padded message length in bytes; 13-15 is left * padded client id
    def gen_header(self, message):
        msg = self.dumps(message)
        return str(len(msg)).ljust(12, "*") + str(self.id).rjust(3, "*")


if __name__ == "__main__":
    # Initialisation stage
    host, port, path, client_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    msg = np.load(path)
    # Connection establishment stage
    client = Client(client_id, host, port)
    client.connect()
    client.sock.setblocking(True)
    init_msg = Message(client.pk, CommStage.CONN_ESTAB)
    client.send(client.dumps(init_msg))
    client.client_num = int(client.recv(10).decode("utf-8"))
    print(client.client_num, "clients in total.")

    # Receive initial model stage
    header = client.recv(1024)
    client.send(b"OK")
    model_message = client.recv(int(client.loads(header)))
    fed_pk_header = client.loads(client.recv(1024))
    client.send(b"OK")
    client.fed_pk = client.loads(client.recv(fed_pk_header))
    model = client.loads(client.loads(model_message).message)
    torch.save(model, "client"+client_id+"_model.pth")
    print("Received model message")

    for _ in range(1):
        model = torch.load("client"+client_id+"_model.pth")
        # Training stage
        # TODO: Training
        # model.fit(data)
        model_grad = client.dumps(client.xcrypt_2d(client.fed_pk.encrypt, np.array(model.weight.data, dtype="float64")))
        model_bias = client.dumps(np.array(list(map(client.fed_pk.encrypt, np.array(model.bias.data, dtype="float64")))))

        # Report stage
        grad_header = Message(len(model_grad), CommStage.REPORT)
        client.send(client.dumps(grad_header))
        ok = client.recv(2)
        if ok.decode("utf-8") == "OK":  # Federator already received the grad len
            client.send(model_grad)
            print("Sent grad")

        bias_header = Message(len(model_bias), CommStage.REPORT)
        client.send(client.dumps(bias_header))
        ok = client.recv(2)
        if ok.decode("utf-8") == "OK":
            client.send(model_bias)
            print("Sent bias")

        # Receive updated model info
        grad_header = int(client.recv(10).decode("utf-8"))
        client.send(b"OK")
        new_grad = client.loads(client.recv(grad_header)).message  # of type np.array with encrypted numbers
        bias_header = int(client.recv(10).decode("utf-8"))
        client.send(b"OK")
        new_bias = client.loads(client.recv(bias_header)).message  # of type np.array with encrypted numbers
        # TODO: Do decryption here
        new_grad = torch.from_numpy(client.xcrypt_2d(client.sk.decrypt, new_grad))
        new_bias = torch.from_numpy(np.array(list(map(client.sk.decrypt, new_bias)), dtype="float64"))
        # Update & save model
        model.weight = torch.nn.Parameter(new_grad / client.client_num)
        model.bias = torch.nn.Parameter(new_bias / client.client_num)
        torch.save(model, "client"+client_id+"_model.pth")
        print("New model saved.")

    # Send END message
    end_msg = Message(b"", CommStage.END)
    client.send(client.dumps(end_msg))
    client.sock.close()
    print("end")
