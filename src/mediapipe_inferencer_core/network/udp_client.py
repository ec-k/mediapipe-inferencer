import socket

class UdpClient:
    def __init__(self, address = None, port = None):
        self.address = address if address is not None else 'localhost'
        self.port = port if port is not None else 9000
        self.server_address = (self.address, self.port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def connect(self):
        self.client_socket.connect(self.server_address)

    def send_text_message(self, msg):
        self.client_socket(msg.encode('utf-8'))

    def send_protobuf_message(self, msg):
        return self.client_socket.send(msg)

    def close(self):
        print("closing socket")
        self.client_socket.close()
        print("done")