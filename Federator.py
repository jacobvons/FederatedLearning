import socket
import selectors
import types
import Message
import sys


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
        self.reporters = []

    def accept_wrapper(self):
        conn, addr = self.sock.accept()
        print('accepted connection from', addr, "\n")
        conn.setblocking(False)
        data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, events, data=data)

    def service_connection(self, key, mask):
        sock = key.fileobj
        data = key.data
        if mask & selectors.EVENT_READ:
            recv_data = sock.recv(1024)  # Should be ready to read
            if recv_data:
                data.outb += recv_data
            else:
                print('closing connection to', data.addr)
                self.reporters.pop()
                self.sel.unregister(sock)
                sock.close()
        if mask & selectors.EVENT_WRITE:
            if data.outb:
                print('echoing', repr(data.outb), 'to', data.addr)
                sent = sock.send(data.outb)  # Should be ready to write
                data.outb = data.outb[sent:]


if __name__ == "__main__":
    host, port = sys.argv[1], int(sys.argv[2])
    fed = Federator(host, port)
    while True:
        try:
            events = fed.sel.select(timeout=None)
            if len(fed.reporters):
                print(fed.reporters)
            for key, mask in events:
                if key.data is None:
                    fed.accept_wrapper(key.fileobj)
                    fed.reporters.append(1)
                else:
                    fed.service_connection(key, mask)
        except:
            fed.sock.close()
            print("Connection closed")
            break
