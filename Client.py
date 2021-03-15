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
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

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

    def update_info(self):
        msg = self.dumps(self.raw_message)
        header = self.gen_header(msg)
        header = self.dumps(header)
        # self.sock.connect((self.host, self.port))
        # self.sock.setblocking(True)
        self.send(header)
        status, client_num = (self.recv(10).decode("utf-8")).split("*")
        client_num = self.dumps(client_num)
        if status == "OK":
            print("Received OK. Sending model parameters.")
            self.send(msg)
            print("Sent. Waiting for update...")
            data, length = self.loads(self.recv(len(msg)+len(client_num)))
            # print(data/int(length))
            self.raw_message = data


if __name__ == "__main__":
    # Initialisation stage
    host, port, path, client_id = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    msg = np.load(path)

    client = Client(client_id, host, port)
    client.connect()
    client.sock.setblocking(True)
    init_msg = Message(b"OK", CommStage.CONN_ESTAB)
    client.send(client.dumps(init_msg))
    print(client.recv(2))

    header = client.recv(1024)
    client.send(b"OK")
    model_message = client.recv(int(client.loads(header)))
    model = client.loads(client.loads(model_message).message)
    torch.save(model, "client"+client_id+"_model.pth")
    print("Received model message")

    # TODO: Training
    # Trained

    # Report stage
    for _ in range(3):
        model = torch.load("client"+client_id+"_model.pth")
        model_grad = client.dumps(model.weight)
        model_bias = client.dumps(model.bias)
        # Info about grad len, also initialises Report process on server
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

    # Send END message
    end_msg = Message(b"", CommStage.END)
    client.send(client.dumps(end_msg))
    print("end")
