# gs_stream_sender.py
import argparse
import glob
import os
import socket
import struct
import time

MAGIC_FILE = b"GSF1"
MAGIC_END  = b"GEND"

def send_one(sock: socket.socket, rel_path: str, abs_path: str):
    rel_path = rel_path.replace("\\", "/")
    path_bytes = rel_path.encode("utf-8")
    size = os.path.getsize(abs_path)

    # magic + (path_len uint16, size uint64) + path + payload
    header = MAGIC_FILE + struct.pack("<HQ", len(path_bytes), size) + path_bytes
    sock.sendall(header)

    t0 = time.perf_counter()
    sent = 0
    with open(abs_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sock.sendall(chunk)
            sent += len(chunk)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[SEND] {rel_path} ({size/1024/1024:.2f} MB) in {dt_ms:.1f} ms")
    return sent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--src", required=True, help="pasta raiz do FVV (onde está init_3dgs.ply, NTCs, additional_3dgs)")
    ap.add_argument("--start", type=int, default=0, help="frame inicial (inclusive)")
    ap.add_argument("--end", type=int, default=298, help="frame final (inclusive)")
    ap.add_argument("--fps", type=float, default=0.0, help="se >0, limita envio por frame (~fps). 0 envia o mais rápido possível")
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    init_ply = os.path.join(src, "init_3dgs.ply")
    config_json = os.path.join(src, "NTCs", "config.json")

    if not os.path.isfile(init_ply):
        raise FileNotFoundError(f"missing: {init_ply}")
    if not os.path.isfile(config_json):
        raise FileNotFoundError(f"missing: {config_json}")

    # lista arquivos por frame
    def ntc_path(i): return os.path.join(src, "NTCs", f"NTC_{i:06d}.pth")
    def add_path(i): return os.path.join(src, "additional_3dgs", f"additions_{i:06d}.ply")

    dt = (1.0 / args.fps) if args.fps and args.fps > 0 else 0.0

    total_bytes = 0
    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        print(f"[SEND] connected to {args.host}:{args.port}")
        # 1) base + config
        total_bytes += send_one(sock, "init_3dgs.ply", init_ply)
        total_bytes += send_one(sock, "NTCs/config.json", config_json)

        # 2) frames
        for i in range(args.start, args.end + 1):
            ntc = ntc_path(i)
            add = add_path(i)

            if not os.path.isfile(ntc):
                raise FileNotFoundError(f"missing NTC: {ntc}")
            total_bytes += send_one(sock, f"NTCs/NTC_{i:06d}.pth", ntc)

            if os.path.isfile(add):
                total_bytes += send_one(sock, f"additional_3dgs/additions_{i:06d}.ply", add)

            if dt > 0:
                time.sleep(dt)

        # 3) fim
        sock.sendall(MAGIC_END)

    print(f"[SEND] done. total={total_bytes/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()
