import socket
import pickle


def send_tensor(tensor, host, port):
    """通过 socket 发送序列化张量"""
    data = pickle.dumps(tensor)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # 发送长度头
        s.sendall(len(data).to_bytes(4, byteorder="big"))
        s.sendall(data)
        # 接收确认
        resp = s.recv(4096)
        return pickle.loads(resp)
