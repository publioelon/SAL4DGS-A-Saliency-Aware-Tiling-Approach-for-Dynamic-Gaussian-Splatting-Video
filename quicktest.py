import os
import zipfile
import torch

p = r"C:\tmp\gs_stream_cache\live_session\NTCs\NTC_000000.pth"

print("file:", p)
print("size:", os.path.getsize(p))
print("is_zip:", zipfile.is_zipfile(p))

if zipfile.is_zipfile(p):
    with zipfile.ZipFile(p, "r") as z:
        names = set(z.namelist())
        print("zip entries:", len(names))
        print("has constants.pkl:", "constants.pkl" in names)
        print("has data.pkl:", "data.pkl" in names)
        head = sorted(list(names))[:25]
        print("first 25 names:")
        for n in head:
            print(" ", n)

print("\n--- torch.jit.load test ---")
try:
    m = torch.jit.load(p, map_location="cpu")
    print("jit: OK ->", type(m))
except Exception as e:
    print("jit: FAIL ->", type(e).__name__, e)

print("\n--- torch.load test ---")
try:
    x = torch.load(p, map_location="cpu")
    print("torch.load: OK ->", type(x))
    if hasattr(x, "keys"):
        print("keys sample:", list(x.keys())[:10])
except Exception as e:
    print("torch.load: FAIL ->", type(e).__name__, e)
