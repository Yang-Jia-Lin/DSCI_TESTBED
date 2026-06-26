import pickle
import socket


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Connection closed before payload completed")
        chunks.extend(chunk)
    return bytes(chunks)


def send_tensor(tensor, host, port, timeout_s=120):
    """Send a length-prefixed pickle payload and wait for a length-prefixed response."""
    data = pickle.dumps(tensor, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[send_tensor] connect {host}:{port}, payload={len(data)} bytes")
    with socket.create_connection((host, port), timeout=float(timeout_s)) as s:
        s.settimeout(float(timeout_s))
        s.sendall(len(data).to_bytes(4, byteorder="big"))
        s.sendall(data)
        print(f"[send_tensor] sent {host}:{port}, waiting response")
        response_len = int.from_bytes(_recv_exact(s, 4), "big")
        response_bytes = _recv_exact(s, response_len)
        print(
            f"[send_tensor] received {host}:{port}, "
            f"response={len(response_bytes)} bytes"
        )
        return pickle.loads(response_bytes)
