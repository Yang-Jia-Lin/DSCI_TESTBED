import socket
import pickle


def send_tensor(tensor, host, port):
    """通过 socket 发送序列化张量"""
    data = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        # 发送长度头
        s.sendall(len(data).to_bytes(4, byteorder="big"))
        s.sendall(data)
        # 循环接收，直到对端关闭连接（Cloud 以 conn.close() 表示响应结束）
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        return pickle.loads(b"".join(chunks))
