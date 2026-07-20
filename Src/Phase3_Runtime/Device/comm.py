import pickle
import socket
import time

from Src.Phase3_Runtime.Shared.tensor_codec import (
    prepare_for_transport,
    restore_from_transport,
)


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Connection closed before payload completed")
        chunks.extend(chunk)
    return bytes(chunks)


_ACK_REQUEST_MASK = 1 << 31
_MAX_PAYLOAD_BYTES = _ACK_REQUEST_MASK - 1


def send_tensor(tensor, host, port, timeout_s=120, *, measure_upload=False):
    """Send a length-prefixed pickle payload and wait for a length-prefixed response."""
    data = pickle.dumps(prepare_for_transport(tensor), protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > _MAX_PAYLOAD_BYTES:
        raise ValueError("Transport payload exceeds the 31-bit framing limit")
    print(f"[send_tensor] connect {host}:{port}, payload={len(data)} bytes")
    with socket.create_connection((host, port), timeout=float(timeout_s)) as s:
        s.settimeout(float(timeout_s))
        header = len(data) | (_ACK_REQUEST_MASK if measure_upload else 0)
        upload_started = time.perf_counter()
        s.sendall(header.to_bytes(4, byteorder="big"))
        s.sendall(data)
        print(f"[send_tensor] sent {host}:{port}, waiting response")
        response_len = int.from_bytes(_recv_exact(s, 4), "big")
        upload_metrics = None
        if measure_upload:
            if response_len != 0:
                raise ConnectionError("Peer did not return the requested upload ACK")
            acknowledged_bytes = int.from_bytes(_recv_exact(s, 8), "big")
            elapsed_s = time.perf_counter() - upload_started
            if acknowledged_bytes != len(data):
                raise ConnectionError(
                    f"Upload ACK length mismatch: sent={len(data)}, ack={acknowledged_bytes}"
                )
            upload_metrics = {
                "payload_bytes": int(acknowledged_bytes),
                "elapsed_s": float(elapsed_s),
                "bw_mbps": float(acknowledged_bytes * 8.0 / elapsed_s / 1e6),
                "observed_at": float(time.time()),
            }
            response_len = int.from_bytes(_recv_exact(s, 4), "big")
        response_bytes = _recv_exact(s, response_len)
        print(
            f"[send_tensor] received {host}:{port}, "
            f"response={len(response_bytes)} bytes"
        )
        response = restore_from_transport(pickle.loads(response_bytes))
        if measure_upload:
            return response, upload_metrics
        return response
