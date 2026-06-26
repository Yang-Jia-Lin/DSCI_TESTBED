"""Concurrent length-prefixed pickle transport used by fixed worker servers."""

from __future__ import annotations

import pickle
import socket
import threading
import traceback


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = conn.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Connection closed before payload completed")
        chunks.extend(chunk)
    return bytes(chunks)


def serve_requests(host: str, port: int, handler, *, backlog: int = 128) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(backlog)
        while True:
            conn, _addr = server.accept()
            threading.Thread(
                target=_handle_connection, args=(conn, handler, _addr), daemon=True
            ).start()


def _handle_connection(conn: socket.socket, handler, addr) -> None:
    with conn:
        try:
            length = int.from_bytes(_recv_exact(conn, 4), "big")
            print(f"[socket_server] accepted {addr}, payload={length} bytes")
            payload = pickle.loads(_recv_exact(conn, length))
            print(f"[socket_server] handling {addr}")
            response = handler(payload)
            print(f"[socket_server] handled {addr}")
        except Exception as exc:
            traceback.print_exc()
            response = {"status": "error", "message": str(exc)}
        response_bytes = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        conn.sendall(len(response_bytes).to_bytes(4, byteorder="big"))
        conn.sendall(response_bytes)
        print(f"[socket_server] responded {addr}, response={len(response_bytes)} bytes")
