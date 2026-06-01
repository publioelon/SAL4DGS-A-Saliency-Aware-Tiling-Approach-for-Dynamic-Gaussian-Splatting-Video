import argparse, os, socket, struct, time, re
from pathlib import Path

CHUNK = 1024 * 1024

NTC_RE = re.compile(r"NTC_(\d{6})\.pth$", re.I)
ADD_RE = re.compile(r"additions_(\d{6})\.ply$", re.I)

def send_exact(sock, data): sock.sendall(data)

def send_file(sock, rel_path: str, abs_path: str):
    rel_path = rel_path.replace("\\", "/")
    pb = rel_path.encode("utf-8")
    size = os.path.getsize(abs_path)
    send_exact(sock, struct.pack("!I", len(pb)))
    send_exact(sock, pb)
    send_exact(sock, struct.pack("!Q", size))

    t0 = time.perf_counter()
    sent = 0
    with open(abs_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            send_exact(sock, chunk)
            sent += len(chunk)
    dt = (time.perf_counter() - t0) * 1000.0
    mb = sent / (1024.0 * 1024.0)
    print(f"[SEND] {rel_path} ({mb:.2f} MB) in {dt:.1f} ms")

def send_end(sock):
    pb = b"END"
    send_exact(sock, struct.pack("!I", len(pb)))
    send_exact(sock, pb)
    send_exact(sock, struct.pack("!Q", 0))

def wait_stable(path: Path, stable_ms=0, timeout_s=30):
    t0 = time.time()
    last = -1
    last_change = time.time()
    while True:
        # se o arquivo nunca aparecer, respeita timeout
        if time.time() - t0 > timeout_s:
            return False

        if not path.exists():
            time.sleep(0.05)
            continue

        sz = path.stat().st_size
        if sz != last:
            last = sz
            last_change = time.time()
        if (time.time() - last_change) * 1000.0 >= stable_ms:
            return True
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5001)
    ap.add_argument("--root", "--src", dest="root", required=True,
                help="FVV root (contains init_3dgs.ply, NTCs/, additional_3dgs/)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)  # opcional: encerra em um frame
    ap.add_argument("--poll_ms", type=int, default=50)
    args = ap.parse_args()

    root = Path(args.root)
    init_ply = root / "init_3dgs.ply"
    ntc_dir = root / "NTCs"
    cfg = ntc_dir / "config.json"

    add_dir = root / "additional_3dgs"
    if not add_dir.exists():
        alt = root / "additional_3dgs_OFF"
        if alt.exists():
            add_dir = alt

    # envia init/config quando estiverem prontos
    if not wait_stable(init_ply): raise FileNotFoundError(init_ply)
    if not wait_stable(cfg): raise FileNotFoundError(cfg)

    with socket.create_connection((args.host, args.port), timeout=10) as s:
        print(f"[SENDER] connected to {args.host}:{args.port}")
        send_file(s, "init_3dgs.ply", str(init_ply))
        send_file(s, "NTCs/config.json", str(cfg))

        sent_ntc = set()
        sent_add = set()

        i = args.start
        while True:
            if args.end >= 0 and i > args.end:
                break

            ntc_path = ntc_dir / f"NTC_{i:06d}.pth"
            add_path = add_dir / f"additions_{i:06d}.ply"

            # manda NTC assim que existir e estiver estável
            if i not in sent_ntc and ntc_path.exists():
                if wait_stable(ntc_path, stable_ms=0, timeout_s=300):
                    send_file(s, f"NTCs/{ntc_path.name}", str(ntc_path))
                    sent_ntc.add(i)

            # manda additions assim que existir e estiver estável
            if i not in sent_add and add_path.exists():
                if wait_stable(add_path, stable_ms=0, timeout_s=300):
                    send_file(s, f"{add_dir.name}/{add_path.name}", str(add_path))
                    sent_add.add(i)

            # só avança quando os dois (ou pelo menos NTC) já foram enviados:
            # (recomendado: exigir NTC, additions pode ser opcional dependendo do pipeline)
            if i in sent_ntc and (add_path.exists() == False or i in sent_add):
                i += 1
            else:
                time.sleep(args.poll_ms / 1000.0)

        send_end(s)

    print("[SENDER] done")

if __name__ == "__main__":
    main()
