import socket
import pickle
import sys
import numpy as np
import phe


class Client:

    def __init__(self, id, host, port, raw_message):
        self.id = id
        self.host = host
        self.port = port
        self.raw_message = raw_message
        self.pk, self.sk = phe.paillier.generate_paillier_keypair()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def send(self, message):
        self.sock.send(message)

    def recv(self, size):
        message = self.sock.recv(size)  # In bytes
        return message

    def xcrypt_2d(self, xcrypt):
        m, n = self.raw_message.shape
        output = []
        for i in range(0, m):
            output.append(np.array(list(map(xcrypt, self.raw_message[i]))))
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

    def update_communication(self):
        msg = self.dumps(self.raw_message)
        header = self.gen_header(msg)
        header = self.dumps(header)
        self.sock.connect((self.host, self.port))
        self.sock.setblocking(True)
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

    client = Client(client_id, host, port, msg)
    # client.raw_message = client.xcrypt_2d(client.pk.encrypt)
    client.update_communication()
    # client.raw_message = client.xcrypt_2d(client.sk.decrypt)
    print(client.raw_message)