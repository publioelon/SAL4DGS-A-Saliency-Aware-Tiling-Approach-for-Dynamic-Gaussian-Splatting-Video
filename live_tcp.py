import os
import re
import json
import time
import socket
import struct
import shutil
import threading
import queue
from pathlib import Path
from collections import OrderedDict

import util_gau

try:
    import torch
except Exception:
    torch = None

try:
    import tinycudann as tcnn
except Exception:
    tcnn = None

from NTC import NeuralTransformationCache


def infer_total_frames(fvv_root: str) -> int:
    root = Path(fvv_root)
    ntc_dir = root / "NTCs"
    ntcs = sorted(ntc_dir.glob("NTC_*.pth"))
    return max(1, len(ntcs) + 1)


# Bigger chunks = fewer syscalls
_TCP_CHUNK = 4 * 1024 * 1024

_NTC_RE = re.compile(r"^NTCs[\\/]+NTC_(\d+)\.pth$", re.IGNORECASE)
# Accept both folder names (sender may use OFF fallback)
_ADD_RE = re.compile(r"^(additional_3dgs|additional_3dgs_OFF)[\\/]+additions_(\d+)\.ply$", re.IGNORECASE)


def _recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def _safe_join(root: Path, rel: str) -> Path:
    """
    Secure join: prevents path traversal.
    Uses Path.relative_to() instead of startswith() (which can be bypassed).
    """
    rel = rel.replace("\\", "/").lstrip("/")
    rel = os.path.normpath(rel)
    if rel.startswith("..") or rel.startswith("../") or rel.startswith("..\\"):
        raise ValueError(f"invalid relative path: {rel}")

    root_res = root.resolve()
    p = (root_res / rel).resolve()
    try:
        p.relative_to(root_res)
    except Exception:
        raise ValueError(f"path escapes root: {rel}")
    return p


class _LazyNTCs:
    """
    List-like object: renderer.NTCs[t] returns a NeuralTransformationCache for frame t.
    Internally uses ONE persistent NTC instance and only swaps weights per index.
    """
    def __init__(self, live_state: "LiveTCPState"):
        self.live = live_state

    def __len__(self):
        # Return something plausible; renderer usually indexes directly.
        if self.live.total_frames_hint > 0:
            return self.live.total_frames_hint
        return max(1, (max(self.live.ntc_paths.keys(), default=-1) + 1))

    def __getitem__(self, idx: int):
        return self.live.get_ntc_for_index(int(idx))


class _LazyAdditions:
    """
    List-like object: renderer.additional_3dgs[t] returns additions for frame t (or None).
    Lazy loads + caches additions. In CUDA mode, always returns CUDA-compatible object.
    """
    def __init__(self, live_state: "LiveTCPState"):
        self.live = live_state

    def __len__(self):
        if self.live.total_frames_hint > 0:
            return self.live.total_frames_hint
        return max(1, (max(self.live.add_paths.keys(), default=-1) + 1))

    def __getitem__(self, idx: int):
        return self.live.get_add_for_index(int(idx))


class LiveTCPState:
    """
    FAST streaming state:
    - Receiver thread writes files to cache + enqueues rel_path
    - UI thread only updates maps and sets list-like providers for renderer
    - NTC is built ONCE and weights swapped per frame (lazy + cached)
    - Additions are lazy-loaded (and cached), not eagerly parsed
    """
    def __init__(
        self,
        cache_root: str,
        total_frames_hint: int = 0,
        autoplay: bool = False,
        verbose: bool = False,
        log_every_n_files: int = 30,
        ntc_state_cache: int = 4,
        add_cache: int = 8,
    ):
        self.cache_root = Path(cache_root)
        self.total_frames_hint = int(total_frames_hint) if total_frames_hint else 0
        self.autoplay = bool(autoplay)

        self.verbose = bool(verbose)
        self.log_every_n_files = max(1, int(log_every_n_files))

        # UI-visible status
        self.status = "idle"
        self.connected = ""
        self.err = ""

        self.files = 0
        self.total_bytes = 0

        self.init_ply_ok = False
        self.config_ok = False

        # Instead of eagerly building lists, we map idx -> path (fast)
        self.ntc_paths = {}   # idx -> abs path
        self.add_paths = {}   # idx -> abs path
        self._any_add_seen = False

        # Renderer-facing list-like objects
        self.ntc_list = _LazyNTCs(self)
        self.add_list = _LazyAdditions(self)

        self.max_play = -1

        try:
            self._q = queue.SimpleQueue()
        except Exception:
            self._q = queue.Queue()

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        # For NTC creation / bounds
        self.gaussians_loaded = False
        self._xyz_min = None
        self._xyz_max = None
        self._ntc_conf = None
        self._base_idx = None

        # Detect renderer mode
        self._cuda_mode = None

        # One persistent NTC + weight swapping
        self._ntc_single = None
        self._ntc_loaded_idx = None
        self._ntc_lock = threading.Lock()
        self._ntc_state_cache_cap = max(0, int(ntc_state_cache))
        self._ntc_state_cache = OrderedDict()  # idx -> state dict (CPU)

        # Lazy additions cache
        self._add_cache_cap = max(0, int(add_cache))
        self._add_cache = OrderedDict()  # idx -> add_obj (CPU or CUDA)

        # Lightweight prefetch (optional, safe default ON)
        self._prefetch_q = queue.Queue(maxsize=32)
        self._prefetch_inflight = set()
        self._prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._prefetch_thread.start()

        # receiver stats
        self._last_log_files = 0

    def request_stop(self):
        self._stop.set()

    # ---------- Receiver (fast I/O) ----------

    def start_receiver(self, bind_host: str, port: int, clear_cache: bool):
        if clear_cache and self.cache_root.exists():
            shutil.rmtree(self.cache_root, ignore_errors=True)

        (self.cache_root / "NTCs").mkdir(parents=True, exist_ok=True)
        (self.cache_root / "additional_3dgs").mkdir(parents=True, exist_ok=True)
        (self.cache_root / "additional_3dgs_OFF").mkdir(parents=True, exist_ok=True)

        def _run():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
            except Exception:
                pass

            srv.bind((bind_host, port))
            srv.listen(1)

            with self._lock:
                self.status = f"listening {bind_host}:{port}"
                self.err = ""

            if self.verbose:
                print(f"[RECV] listening on {bind_host}:{port}")

            srv.settimeout(0.5)
            conn = None
            addr = None
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                    break
                except socket.timeout:
                    continue

            if self._stop.is_set():
                try:
                    srv.close()
                except Exception:
                    pass
                return

            with self._lock:
                self.status = "connected"
                self.connected = f"{addr[0]}:{addr[1]}"
                self.err = ""

            if self.verbose:
                print(f"[RECV] connected from {addr[0]}:{addr[1]}")

            try:
                with conn:
                    try:
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception:
                        pass
                    try:
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
                    except Exception:
                        pass
                    conn.settimeout(1.0)

                    buf = bytearray(_TCP_CHUNK)

                    while not self._stop.is_set():
                        try:
                            hdr = _recvall(conn, 4)
                        except socket.timeout:
                            continue
                        except EOFError:
                            break

                        (path_len,) = struct.unpack("!I", hdr)
                        if path_len <= 0 or path_len > 1024 * 16:
                            raise RuntimeError(f"bad path_len={path_len}")

                        rel_path = _recvall(conn, path_len).decode("utf-8", errors="replace")
                        (size,) = struct.unpack("!Q", _recvall(conn, 8))

                        if rel_path in ("END", "__END__"):
                            break

                        out_path = _safe_join(self.cache_root, rel_path)
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = out_path.with_suffix(out_path.suffix + ".part")

                        remaining = size
                        written = 0

                        t0 = time.perf_counter()
                        try:
                            with open(tmp_path, "wb", buffering=1024 * 1024) as f:
                                mv = memoryview(buf)
                                while remaining > 0:
                                    n = _TCP_CHUNK if remaining >= _TCP_CHUNK else int(remaining)
                                    view = mv[:n]
                                    try:
                                        got = conn.recv_into(view)
                                    except socket.timeout:
                                        continue
                                    if got <= 0:
                                        raise EOFError("socket closed mid-file")
                                    f.write(view[:got])
                                    remaining -= got
                                    written += got
                            os.replace(tmp_path, out_path)
                        finally:
                            if tmp_path.exists() and not out_path.exists():
                                try:
                                    tmp_path.unlink()
                                except Exception:
                                    pass

                        dt_ms = (time.perf_counter() - t0) * 1000.0

                        with self._lock:
                            self.files += 1
                            self.total_bytes += written

                        # logging throttled (printing is slow)
                        if self.verbose and (self.files % self.log_every_n_files == 0):
                            mb = written / (1024.0 * 1024.0)
                            print(f"[RECV] #{self.files} last={rel_path} ({mb:.2f} MB) {dt_ms:.1f} ms")

                        self._q.put(rel_path)

            except Exception as e:
                with self._lock:
                    self.status = "error"
                    self.err = repr(e)
                if self.verbose:
                    print("[RECV] ERROR:", repr(e))

            with self._lock:
                if self.status != "error":
                    self.status = "done"

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    # ---------- NTC build / swap (FAST) ----------

    def _torch_device(self) -> str:
        if torch is None:
            raise RuntimeError("torch not available")
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_ntc_conf(self):
        if self._ntc_conf is not None:
            return
        cfg_path = self.cache_root / "NTCs" / "config.json"
        if not cfg_path.exists():
            raise RuntimeError("NTCs/config.json not received yet")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self._ntc_conf = json.load(f)
        if ("encoding" not in self._ntc_conf) or ("network" not in self._ntc_conf):
            raise RuntimeError("Invalid NTC config.json (missing 'encoding'/'network')")
        self.config_ok = True

    def _ensure_bounds(self, renderer):
        if self._xyz_min is not None and self._xyz_max is not None:
            return
        if not hasattr(renderer, "gaussians") or renderer.gaussians is None:
            raise RuntimeError("renderer.gaussians not ready for bounds")
        if not hasattr(renderer.gaussians, "get_xyz_bound"):
            raise RuntimeError("renderer.gaussians missing get_xyz_bound()")
        xyz_min, xyz_max = renderer.gaussians.get_xyz_bound()
        self._xyz_min = xyz_min
        self._xyz_max = xyz_max

    def _ensure_cuda_mode(self, renderer):
        if self._cuda_mode is not None:
            return
        cuda_mode = False
        if torch is not None and hasattr(renderer, "gaussians") and hasattr(renderer.gaussians, "xyz"):
            if isinstance(renderer.gaussians.xyz, torch.Tensor):
                cuda_mode = bool(renderer.gaussians.xyz.is_cuda)
        self._cuda_mode = cuda_mode

    def _ensure_ntc_single(self, renderer):
        if self._ntc_single is not None:
            return
        if torch is None:
            raise RuntimeError("torch not available")
        if tcnn is None:
            raise RuntimeError("tinycudann not available")

        dev = self._torch_device()
        if dev != "cuda":
            raise RuntimeError("NTC requires CUDA (tinycudann). torch.cuda.is_available() is False")

        self._ensure_ntc_conf()
        self._ensure_bounds(renderer)

        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=8,
            encoding_config=self._ntc_conf["encoding"],
            network_config=self._ntc_conf["network"],
        ).cuda()

        ntc = NeuralTransformationCache(model, self._xyz_min, self._xyz_max).cuda()
        ntc.eval()

        self._ntc_single = ntc
        self._ntc_loaded_idx = None

    def _cache_put(self, od: OrderedDict, key, val, cap: int):
        od[key] = val
        od.move_to_end(key)
        while cap > 0 and len(od) > cap:
            od.popitem(last=False)

    def _load_state_dict_fast(self, path: str):
        # Try weights_only if available (torch>=2), else normal
        if torch is None:
            raise RuntimeError("torch not available")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def _prefetch_loop(self):
        while True:
            idx = self._prefetch_q.get()
            if idx is None:
                return
            try:
                # NTC prefetch
                p = self.ntc_paths.get(idx)
                if p and self._ntc_state_cache_cap > 0:
                    if idx not in self._ntc_state_cache:
                        sd = self._load_state_dict_fast(p)
                        self._cache_put(self._ntc_state_cache, idx, sd, self._ntc_state_cache_cap)
                # Additions prefetch (CPU only; CUDA conversion happens on demand in UI thread)
                ap = self.add_paths.get(idx)
                if ap and self._add_cache_cap > 0 and idx not in self._add_cache:
                    add_cpu = util_gau.load_ply(ap)
                    self._cache_put(self._add_cache, idx, add_cpu, self._add_cache_cap)
            except Exception:
                pass
            finally:
                try:
                    self._prefetch_inflight.discard(idx)
                except Exception:
                    pass

    def _request_prefetch(self, idx: int):
        if idx < 0:
            return
        if idx in self._prefetch_inflight:
            return
        if self._prefetch_q.full():
            return
        self._prefetch_inflight.add(idx)
        try:
            self._prefetch_q.put_nowait(idx)
        except Exception:
            self._prefetch_inflight.discard(idx)

    def get_ntc_for_index(self, idx: int):
        """
        Called by renderer during draw: returns the NTC object for a given timestep.
        This is the hot path: must be fast.
        """
        if not (self.init_ply_ok and self.config_ok):
            return None
        path = self.ntc_paths.get(idx)
        if path is None:
            return None

        with self._ntc_lock:
            try:
                self._ensure_ntc_single(self._renderer_ref)
            except Exception as e:
                with self._lock:
                    self.err = f"NTC init error: {repr(e)}"
                return None

            if self._ntc_loaded_idx != idx:
                # state dict from cache or disk
                state = self._ntc_state_cache.get(idx)
                if state is None:
                    try:
                        state = self._load_state_dict_fast(path)
                    except Exception as e:
                        with self._lock:
                            self.err = f"load NTC_{idx:06d}.pth: {repr(e)}"
                        return None
                    if self._ntc_state_cache_cap > 0:
                        self._cache_put(self._ntc_state_cache, idx, state, self._ntc_state_cache_cap)

                try:
                    self._ntc_single.load_state_dict(state, strict=False)
                    self._ntc_loaded_idx = idx
                except Exception as e:
                    with self._lock:
                        self.err = f"apply NTC_{idx:06d}.pth: {repr(e)}"
                    return None

                # Prefetch next frame (sequential playback)
                self._request_prefetch(idx + 1)

            return self._ntc_single

    def get_add_for_index(self, idx: int):
        """
        Called by renderer: returns additions for timestep idx, or None.
        Lazy loads + caches. In CUDA mode, never returns CPU-only data.
        """
        if not self.init_ply_ok:
            return None

        path = self.add_paths.get(idx)
        if path is None:
            return None

        # cached?
        add_obj = self._add_cache.get(idx)
        if add_obj is not None:
            # Prefetch next (helps sequential)
            self._request_prefetch(idx + 1)
            # If CUDA mode and cache contains CPU object, convert on demand
            if self._cuda_mode:
                try:
                    from renderer_cuda import gaus_cuda_from_cpu
                    if not (torch is not None and isinstance(getattr(self._renderer_ref.gaussians, "xyz", None), torch.Tensor)):
                        return None
                    # Convert CPU->CUDA (one-time)
                    add_cuda = gaus_cuda_from_cpu(add_obj)
                    self._cache_put(self._add_cache, idx, add_cuda, self._add_cache_cap)
                    return add_cuda
                except Exception as e:
                    with self._lock:
                        self.err = f"CUDA convert additions_{idx:06d}.ply: {repr(e)}"
                    return None
            return add_obj

        # Not cached: load now (can be heavy)
        try:
            add_cpu = util_gau.load_ply(path)
        except Exception as e:
            with self._lock:
                self.err = f"load additions_{idx:06d}.ply: {repr(e)}"
            return None

        if self._add_cache_cap > 0:
            self._cache_put(self._add_cache, idx, add_cpu, self._add_cache_cap)

        # Prefetch next
        self._request_prefetch(idx + 1)

        if self._cuda_mode:
            try:
                from renderer_cuda import gaus_cuda_from_cpu
                add_cuda = gaus_cuda_from_cpu(add_cpu)
                if self._add_cache_cap > 0:
                    self._cache_put(self._add_cache, idx, add_cuda, self._add_cache_cap)
                return add_cuda
            except Exception as e:
                with self._lock:
                    self.err = f"CUDA convert additions_{idx:06d}.ply: {repr(e)}"
                return None

        return add_cpu

    # ---------- UI thread ingest (FAST) ----------

    def process_new_files(self, renderer, camera):
        """
        Called every UI tick.
        Goal: be extremely cheap.
        """
        # Keep a reference so lazy getters can access renderer
        self._renderer_ref = renderer

        # process queue with a small time budget to avoid UI stutter
        t_budget_ms = 3.0
        t0 = time.perf_counter()

        while True:
            # time budget
            if (time.perf_counter() - t0) * 1000.0 > t_budget_ms:
                break

            try:
                rel_path = self._q.get_nowait()
            except Exception:
                break

            abs_path = (self.cache_root / rel_path).resolve()
            rel_norm = rel_path.replace("\\", "/")

            if rel_norm == "init_3dgs.ply":
                try:
                    gaussians = util_gau.load_ply(str(abs_path))
                    renderer.update_gaussian_data(gaussians)
                    renderer.sort_and_update(camera)

                    self.gaussians_loaded = True
                    self.init_ply_ok = True

                    # bounds must be recomputed from real gaussians
                    self._xyz_min = None
                    self._xyz_max = None

                    # detect renderer mode
                    self._ensure_cuda_mode(renderer)

                    # attach lazy providers
                    renderer.NTCs = self.ntc_list
                    renderer.additional_3dgs = self.add_list

                    # if config already there, create the single NTC now
                    if self.config_ok:
                        try:
                            self._ensure_ntc_single(renderer)
                        except Exception as e:
                            with self._lock:
                                self.err = f"NTC init error: {repr(e)}"

                except Exception as e:
                    with self._lock:
                        self.err = f"load init_3dgs.ply: {repr(e)}"
                continue

            if rel_norm.lower() == "ntcs/config.json":
                try:
                    self._ntc_conf = None
                    self._ensure_ntc_conf()
                    # if init already there, create NTC once now
                    if self.init_ply_ok:
                        try:
                            self._ensure_ntc_single(renderer)
                        except Exception as e:
                            with self._lock:
                                self.err = f"NTC init error: {repr(e)}"
                except Exception as e:
                    with self._lock:
                        self.err = f"load NTCs/config.json: {repr(e)}"
                continue

            m_ntc = _NTC_RE.match(rel_path.replace("/", "\\"))
            if m_ntc:
                idx_real = int(m_ntc.group(1))
                if self._base_idx is None:
                    self._base_idx = idx_real
                idx = idx_real - self._base_idx
                if idx < 0:
                    idx = idx_real
                self.ntc_paths[idx] = str(abs_path)
                # Prefetch soon-ish
                self._request_prefetch(idx)
                continue

            m_add = _ADD_RE.match(rel_path.replace("/", "\\"))
            if m_add:
                idx_real = int(m_add.group(2))
                if self._base_idx is None:
                    self._base_idx = idx_real
                idx = idx_real - self._base_idx
                if idx < 0:
                    idx = idx_real
                self.add_paths[idx] = str(abs_path)
                self._any_add_seen = True
                self._request_prefetch(idx)
                continue

        # Update max_play based on contiguous availability (super fast)
        k = 0
        while k in self.ntc_paths:
            k += 1

        if self._any_add_seen:
            a = 0
            while a in self.add_paths:
                a += 1
            self.max_play = min(k, a) - 1
        else:
            self.max_play = k - 1

        with self._lock:
            if (
                self.status == "connected"
                and self.max_play >= 0
                and self.init_ply_ok
                and self.config_ok
            ):
                self.status = "streaming"

    def ui_snapshot(self):
        with self._lock:
            gb = self.total_bytes / (1024.0 * 1024.0 * 1024.0)

        # show contiguous ready counts (more meaningful than total received)
        k = 0
        while k in self.ntc_paths:
            k += 1
        a = 0
        while a in self.add_paths:
            a += 1

        return {
            "cache": str(self.cache_root),
            "status": self.status,
            "connected": self.connected,
            "err": self.err,
            "init_ply": self.init_ply_ok,
            "config": self.config_ok,
            "ready_ntc": k,
            "ready_add": a,
            "max_play": self.max_play,
            "files": self.files,
            "gb": gb,
            "base_idx": self._base_idx,
        }
