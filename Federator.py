import socket
import selectors
import types
import Message
import sys
import pickle


class Federator:

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.sel = selectors.DefaultSelector()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen()
        print('listening on', (host, port))
        self.sock.setblocking(False)
        self.sel.register(self.sock, selectors.EVENT_READ, data=None)
        self.messages = []

    def accept_wrapper(self):
        conn, addr = self.sock.accept()
        print('accepted connection from', addr, "\n")
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
        event_actions = selectors.EVENT_READ | selectors.EVENT_WRITE  # We want both read and write to client
        self.sel.register(conn, event_actions, data=data)

    def service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                data.outb += recv_data
            else:
                print('closing connection to', data.addr)
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if data.outb:
                print('echoing', repr(data.outb))
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]
                print(sent)


def accept_wrapper(sock, sel):
    conn, addr = sock.accept()
    print('accepted connection from', addr, "\n")
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'', id="")
    event_actions = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, event_actions, data=data)


if __name__ == "__main__":
    host, port = sys.argv[1], int(sys.argv[2])
    sel = selectors.DefaultSelector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen()
    print("Listening on", host+":"+str(port))
    sock.setblocking(False)
    sel.register(sock, selectors.EVENT_READ, data=None)
    while True:
        events = sel.select(timeout=None)
        for key, mask in events:
            if key.data is None:
                accept_wrapper(key.fileobj, sel)
            else:  # Service this connection
                sock = key.fileobj
                data = key.data
                if mask & selectors.EVENT_READ:
                    try:
                        recv_header = sock.recv(1024)  # Ready to read the header.
                        msg = pickle.loads(recv_header)
                        msg_size, client_id = msg[:13].strip("*"), msg[13:].strip("*")
                        print("Preparing to receive", msg_size, "bytes from client", client_id)
                        sock.setblocking(True)
                        sock.send(b"OK")
                        recv_data = sock.recv(int(msg_size))

                        if recv_data:
                            data.outb += recv_data
                            data.id = client_id

                    except Exception as e:
                        print(e)
                        print('closing connection to', data.addr)
                        # sel.unregister(sock)
                        # sock.close()

                if mask & selectors.EVENT_WRITE:
                    if data.outb:
                        print('echoing data to', data.addr, "(client", data.id+")")
                        sent = sock.send(data.outb)  # Should be ready to write
                        data.outb = data.outb[sent:]
                        sel.unregister(sock)
