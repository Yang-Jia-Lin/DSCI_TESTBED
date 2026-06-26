import pickle
import socket


def send_tensor(tensor, host, port, timeout_s=120):
    """Send a length-prefixed pickle payload and wait for the peer response."""
    data = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[send_tensor] connect {host}:{port}, payload={len(data)} bytes")
    with socket.create_connection((host, port), timeout=float(timeout_s)) as s:
        s.settimeout(float(timeout_s))
        s.sendall(len(data).to_bytes(4, byteorder="big"))
        s.sendall(data)
        print(f"[send_tensor] sent {host}:{port}, waiting response")
        chunks = []
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
        response_bytes = b"".join(chunks)
        print(
            f"[send_tensor] received {host}:{port}, "
            f"response={len(response_bytes)} bytes"
        )
        return pickle.loads(response_bytes)
