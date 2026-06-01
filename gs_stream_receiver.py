# gs_stream_receiver.py
import argparse
import os
import socket
import struct
import time

MAGIC_FILE = b"GSF1"   # registro de arquivo
MAGIC_END  = b"GEND"   # fim da transmissão

def recvall(sock: socket.socket, n: int) -> bytes:
    """Lê exatamente n bytes do socket (ou levanta EOFError)."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError("socket closed while receiving")
        data.extend(chunk)
    return bytes(data)

def safe_join(root: str, rel: str) -> str:
    # Normaliza separadores e impede path traversal
    rel = rel.replace("\\", "/").lstrip("/")
    full = os.path.normpath(os.path.join(root, rel))
    root_norm = os.path.normpath(root)
    if not full.startswith(root_norm):
        raise ValueError(f"Unsafe path: {rel}")
    return full

def recv_files(conn: socket.socket, dst_root: str):
    os.makedirs(dst_root, exist_ok=True)
    print(f"[RECV] saving into: {dst_root}")

    total_bytes = 0
    file_count = 0

    while True:
        magic = recvall(conn, 4)
        if magic == MAGIC_END:
            print("[RECV] got END")
            break
        if magic != MAGIC_FILE:
            raise RuntimeError(f"Bad magic: {magic!r}")

        # header: path_len (uint16) + size (uint64)
        hdr = recvall(conn, 2 + 8)
        path_len, size = struct.unpack("<HQ", hdr)

        rel_path = recvall(conn, path_len).decode("utf-8", errors="strict")
        out_path = safe_join(dst_root, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        t0 = time.perf_counter()
        remaining = size
        with open(out_path, "wb") as f:
            while remaining > 0:
                chunk = conn.recv(min(1024 * 1024, remaining))
                if not chunk:
                    raise EOFError("socket closed mid-file")
                f.write(chunk)
                remaining -= len(chunk)

        dt_ms = (time.perf_counter() - t0) * 1000.0
        total_bytes += size
        file_count += 1
        print(f"[RECV] {rel_path}  ({size/1024/1024:.2f} MB)  in {dt_ms:.1f} ms")

    print(f"[RECV] done. files={file_count}, total={total_bytes/1024/1024:.2f} MB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind", default="127.0.0.1", help="IP para escutar (ex.: 0.0.0.0)")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--dst", default=r"C:\tmp\gs_stream_cache\live_session")
    args = ap.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.bind, args.port))
        s.listen(1)
        print(f"[RECV] listening on {args.bind}:{args.port}")
        conn, addr = s.accept()
        with conn:
            print(f"[RECV] connected from {addr[0]}:{addr[1]}")
            recv_files(conn, args.dst)

if __name__ == "__main__":
    main()
