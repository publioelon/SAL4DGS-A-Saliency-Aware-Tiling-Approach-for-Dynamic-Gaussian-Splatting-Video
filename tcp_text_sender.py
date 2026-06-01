# tcp_text_sender.py
import argparse
import socket
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--text", default="hello")
    args = ap.parse_args()

    dt = 1.0 / max(args.fps, 0.001)

    with socket.create_connection((args.host, args.port), timeout=5) as s:
        print(f"[SENDER] connected to {args.host}:{args.port}")
        for i in range(args.count):
            send_ns = time.time_ns()  # relógio de parede (OK p/ localhost)
            payload = f"{i}|{send_ns}|{args.text} #{i}\n"
            s.sendall(payload.encode("utf-8"))
            time.sleep(dt)

    print("[SENDER] done")

if __name__ == "__main__":
    main()
