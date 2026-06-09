import socket
import pickle


def receive_tensor(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)
    print(f"云特征监听 {host}:{port}")
    conn, addr = s.accept()
    s.close()

    try:
        length_bytes = conn.recv(4)
        if not length_bytes:
            conn.close()
            return None, None
        length = int.from_bytes(length_bytes, 'big')
        data = b''
        while len(data) < length:
            chunk = conn.recv(length - len(data))
            if not chunk:
                break
            data += chunk
        payload = pickle.loads(data)
        return payload, conn
    except Exception as e:
        print("云接收错误:", e)
        conn.close()
        return None, None
