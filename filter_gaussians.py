#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import csv
import json
import math
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    import tinycudann as tcnn
except Exception:
    tcnn = None

from NTC import NeuralTransformationCache
from util_gau import load_ply


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
cv2.setNumThreads(1)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
NTC_RE = re.compile(r"NTC_(\d+)\.pth$", re.IGNORECASE)
ADD_RE = re.compile(r"additions_(\d+)\.ply$", re.IGNORECASE)

DEFAULT_TILE_ROWS = 5
DEFAULT_TILE_COLS = 9
DEFAULT_RED_THRESH = 0.22

K3 = np.ones((3, 3), dtype=np.uint8)


# ============================================================
# Basic helpers
# ============================================================

def parse_vec3(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected vec3 as x,y,z but got: {s}")
    return np.asarray(vals, dtype=np.float32)


def to_numpy_xyz(xyz_obj) -> np.ndarray:
    if isinstance(xyz_obj, np.ndarray):
        return xyz_obj.astype(np.float32, copy=False)

    if torch.is_tensor(xyz_obj):
        return xyz_obj.detach().float().cpu().numpy()

    return np.asarray(xyz_obj, dtype=np.float32)


def to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x

    elif torch.is_tensor(x):
        arr = x.detach().float().cpu().numpy()

    else:
        arr = np.asarray(x)

    return arr.astype(np.float32).reshape(-1)


def normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)

    vals = x[finite]
    mn = float(vals.min())
    mx = float(vals.max())

    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)

    y = (x - mn) / (mx - mn)
    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def list_images_sorted(folder: str):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")

    files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)]
    files.sort()
    return files


def adjust_dimensions(img, rows, cols):
    h, w = img.shape[:2]
    return img[: (h // rows) * rows, : (w // cols) * cols]


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero-length vector")
    return v / n


# ============================================================
# Tiling
# ============================================================

def compute_tile_classes_from_saliency(
    sal_map_u8,
    tile_rows=DEFAULT_TILE_ROWS,
    tile_cols=DEFAULT_TILE_COLS,
    red_thresh=DEFAULT_RED_THRESH,
):
    """
    Computes tile classes:
        red   = average saliency > threshold
        green = adjacent to red
        black = remaining tiles
    """
    sal_adj = adjust_dimensions(sal_map_u8, tile_rows, tile_cols)
    h, w = sal_adj.shape[:2]
    th, tw = h // tile_rows, w // tile_cols

    sal_tiles = sal_adj.reshape(tile_rows, th, tile_cols, tw).transpose(0, 2, 1, 3)
    avg_sal = sal_tiles.mean(axis=(2, 3)) / 255.0

    red_mask = avg_sal > float(red_thresh)
    dilated = cv2.dilate(red_mask.astype(np.uint8), K3, iterations=1).astype(bool)
    green_mask = dilated & (~red_mask)

    klass_map = np.full((tile_rows, tile_cols), "black", dtype=object)
    klass_map[green_mask] = "green"
    klass_map[red_mask] = "red"

    return sal_adj, avg_sal, klass_map


def tile_classes_for_uv(
    uv: np.ndarray,
    valid: np.ndarray,
    klass_map,
    width: int,
    height: int,
    tile_rows: int,
    tile_cols: int,
) -> np.ndarray:
    """
    Returns one tile class per projected Gaussian:
        red, green, black, invalid
    """
    classes = np.full((uv.shape[0],), "invalid", dtype=object)

    if uv.shape[0] == 0:
        return classes

    idx = np.where(valid)[0]
    if idx.size == 0:
        return classes

    tile_w = width / tile_cols
    tile_h = height / tile_rows

    u = uv[idx, 0]
    v = uv[idx, 1]

    tx = np.clip((u / tile_w).astype(np.int32), 0, tile_cols - 1)
    ty = np.clip((v / tile_h).astype(np.int32), 0, tile_rows - 1)

    classes[idx] = np.array([klass_map[r, c] for r, c in zip(ty, tx)], dtype=object)

    return classes


def classify_black_tiles(
    uv: np.ndarray,
    valid: np.ndarray,
    klass_map,
    width: int,
    height: int,
    tile_rows: int,
    tile_cols: int,
):
    black = np.zeros(uv.shape[0], dtype=bool)

    if uv.shape[0] == 0:
        return black

    classes = tile_classes_for_uv(
        uv=uv,
        valid=valid,
        klass_map=klass_map,
        width=width,
        height=height,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )

    black = classes == "black"

    return black


# ============================================================
# Camera projection
# ============================================================

class PerspectiveProjector:
    def __init__(self, cam_pos, cam_target, cam_up, fov_y_deg, width, height):
        self.cam_pos = np.asarray(cam_pos, dtype=np.float32)
        self.cam_target = np.asarray(cam_target, dtype=np.float32)
        self.cam_up = np.asarray(cam_up, dtype=np.float32)

        self.width = int(width)
        self.height = int(height)
        self.fov_y_deg = float(fov_y_deg)

        self.forward = normalize(self.cam_target - self.cam_pos)
        self.right = normalize(np.cross(self.forward, self.cam_up))
        self.true_up = normalize(np.cross(self.right, self.forward))

        fovy = math.radians(self.fov_y_deg)
        self.fy = (self.height * 0.5) / math.tan(max(1e-8, fovy * 0.5))
        self.fx = self.fy * (self.width / self.height)
        self.cx = self.width * 0.5
        self.cy = self.height * 0.5

    def project(self, points_world: np.ndarray):
        pw = np.asarray(points_world, dtype=np.float32)
        rel = pw - self.cam_pos[None, :]

        x_cam = rel @ self.right
        y_cam = rel @ self.true_up
        z_cam = rel @ self.forward

        valid = z_cam > 1e-6
        z = np.maximum(z_cam, 1e-6)

        uv = np.zeros((pw.shape[0], 2), dtype=np.float32)
        uv[:, 0] = self.fx * (x_cam / z) + self.cx
        uv[:, 1] = self.cy - self.fy * (y_cam / z)

        valid &= (
            (uv[:, 0] >= 0.0)
            & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] < self.height)
        )

        return uv, valid, z_cam


# ============================================================
# FVV / 3DGStream loading helpers
# ============================================================

def find_ntc_paths(fvv_root: str):
    ntc_dir = Path(fvv_root) / "NTCs"
    paths = {}

    for p in sorted(ntc_dir.glob("NTC_*.pth")):
        m = NTC_RE.match(p.name)
        if m:
            paths[int(m.group(1))] = str(p)

    return paths


def find_additions_dir(fvv_root: str):
    p = Path(fvv_root) / "additional_3dgs"
    if p.exists():
        return p

    q = Path(fvv_root) / "additional_3dgs_OFF"
    if q.exists():
        return q

    return None


def find_add_paths(fvv_root: str):
    add_dir = find_additions_dir(fvv_root)

    if add_dir is None:
        return {}

    paths = {}

    for p in sorted(add_dir.glob("additions_*.ply")):
        m = ADD_RE.match(p.name)
        if m:
            paths[int(m.group(1))] = str(p)

    return paths


def load_ntc_config(fvv_root: str):
    cfg = Path(fvv_root) / "NTCs" / "config.json"

    if not cfg.exists():
        raise FileNotFoundError(f"NTCs/config.json not found: {cfg}")

    with open(cfg, "r", encoding="utf-8") as f:
        conf = json.load(f)

    if "encoding" not in conf or "network" not in conf:
        raise RuntimeError("Invalid NTC config.json; expected keys 'encoding' and 'network'")

    return conf


def load_state_dict_flexible(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_ntc_model(ntc_conf: dict, xyz_min: np.ndarray, xyz_max: np.ndarray):
    if tcnn is None:
        raise RuntimeError("tinycudann is required for NTC deformation but is not available")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NTC deformation in this script")

    model = tcnn.NetworkWithInputEncoding(
        n_input_dims=3,
        n_output_dims=8,
        encoding_config=ntc_conf["encoding"],
        network_config=ntc_conf["network"],
    ).cuda()

    xyz_min_t = torch.as_tensor(xyz_min, dtype=torch.float32, device="cuda")
    xyz_max_t = torch.as_tensor(xyz_max, dtype=torch.float32, device="cuda")

    ntc = NeuralTransformationCache(model, xyz_min_t, xyz_max_t).cuda()
    ntc.eval()

    return ntc


# ============================================================
# Base Gaussian deformation/projection
# ============================================================

@torch.no_grad()
def compute_base_black_mask_and_projection_for_frame(
    base_xyz_np: np.ndarray,
    ntc_model: NeuralTransformationCache,
    ntc_state_path: str,
    projector: PerspectiveProjector,
    klass_map,
    tile_rows: int,
    tile_cols: int,
    chunk_size: int = 200000,
    return_projection: bool = False,
):
    """
    Computes the framewise black mask for base Gaussians.

    Optionally also returns full projected uv, valid, and depth arrays.
    """
    state = load_state_dict_flexible(ntc_state_path)
    ntc_model.load_state_dict(state, strict=False)
    ntc_model.eval()

    n = base_xyz_np.shape[0]
    out_black = np.zeros(n, dtype=bool)

    uv_chunks = []
    valid_chunks = []
    z_chunks = []

    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)

        xyz_chunk = torch.as_tensor(base_xyz_np[s:e], dtype=torch.float32, device="cuda")
        _, d_xyz, _ = ntc_model(xyz_chunk)

        xyz_t = (xyz_chunk + d_xyz.float()).detach().cpu().numpy()

        uv, valid, z_cam = projector.project(xyz_t)

        out_black[s:e] = classify_black_tiles(
            uv=uv,
            valid=valid,
            klass_map=klass_map,
            width=projector.width,
            height=projector.height,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
        )

        if return_projection:
            uv_chunks.append(uv)
            valid_chunks.append(valid)
            z_chunks.append(z_cam.astype(np.float32))

    if return_projection:
        uv_all = np.concatenate(uv_chunks, axis=0) if uv_chunks else np.zeros((0, 2), np.float32)
        valid_all = np.concatenate(valid_chunks, axis=0) if valid_chunks else np.zeros((0,), bool)
        z_all = np.concatenate(z_chunks, axis=0) if z_chunks else np.zeros((0,), np.float32)

        return out_black, uv_all, valid_all, z_all

    return out_black


def compute_add_black_mask_for_frame(
    add_xyz_np: np.ndarray,
    projector: PerspectiveProjector,
    klass_map,
    tile_rows: int,
    tile_cols: int,
):
    uv, valid, _ = projector.project(add_xyz_np)

    return classify_black_tiles(
        uv=uv,
        valid=valid,
        klass_map=klass_map,
        width=projector.width,
        height=projector.height,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )


# ============================================================
# Debug drawing
# ============================================================

def maybe_draw_debug_points(frame_bgr, uv, black_mask, out_path, max_points=30000):
    vis = frame_bgr.copy()

    idx = np.where(black_mask)[0]

    if idx.size == 0:
        cv2.imwrite(out_path, vis)
        return

    if idx.size > max_points:
        step = max(1, idx.size // max_points)
        idx = idx[::step]

    for i in idx:
        x, y = uv[i]
        cv2.circle(
            vis,
            (int(round(x)), int(round(y))),
            1,
            (0, 0, 255),
            -1,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(out_path, vis)


def draw_tile_grid(frame_bgr, tile_rows, tile_cols):
    vis = frame_bgr.copy()
    h, w = vis.shape[:2]

    for r in range(1, tile_rows):
        y = int(round(r * h / tile_rows))
        cv2.line(vis, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)

    for c in range(1, tile_cols):
        x = int(round(c * w / tile_cols))
        cv2.line(vis, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def draw_selected_gaussian_labels(
    frame_bgr,
    uv_all: np.ndarray,
    valid_all: np.ndarray,
    selected_items: list[dict],
    out_path: str,
    tile_rows: int,
    tile_cols: int,
    draw_grid: bool = True,
):
    """
    Draws labels B01..B10 on a 2D frame using the projected coordinates.
    """
    vis = frame_bgr.copy()

    if draw_grid:
        vis = draw_tile_grid(vis, tile_rows, tile_cols)

    h, w = vis.shape[:2]

    for item in selected_items:
        gid = int(item["id"])
        label = str(item["label"])

        if gid < 0 or gid >= uv_all.shape[0]:
            continue

        if gid >= valid_all.shape[0] or not bool(valid_all[gid]):
            continue

        x, y = uv_all[gid]
        px = int(round(float(x)))
        py = int(round(float(y)))

        if not (0 <= px < w and 0 <= py < h):
            continue

        # Marker
        cv2.circle(vis, (px, py), 7, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (px, py), 9, (0, 0, 0), 2, lineType=cv2.LINE_AA)

        # Label box
        text = label
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.65
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        tx = min(max(px + 10, 0), max(0, w - tw - 8))
        ty = min(max(py - 10, th + 8), max(th + 8, h - baseline - 4))

        cv2.rectangle(
            vis,
            (tx - 4, ty - th - 6),
            (tx + tw + 4, ty + baseline + 4),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            vis,
            text,
            (tx, ty),
            font,
            scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)


# ============================================================
# Visible-center Gaussian selection
# ============================================================

def parse_target_classes(s: str) -> set[str]:
    vals = [x.strip().lower() for x in s.split(",") if x.strip()]
    allowed = {"red", "green", "black", "invalid", "any", "all"}

    for v in vals:
        if v not in allowed:
            raise ValueError(f"Invalid target class '{v}'. Allowed: {sorted(allowed)}")

    if not vals:
        return {"red", "green"}

    if "any" in vals or "all" in vals:
        return {"red", "green", "black"}

    return set(vals)


def select_visible_center_base_gaussians(
    uv_all: np.ndarray,
    valid_all: np.ndarray,
    z_all: np.ndarray,
    base_opacity: np.ndarray,
    klass_map,
    projector: PerspectiveProjector,
    tile_rows: int,
    tile_cols: int,
    count: int = 10,
    target_classes: set[str] | None = None,
    center_x_min: float = 0.40,
    center_x_max: float = 0.60,
    center_y_min: float = 0.35,
    center_y_max: float = 0.65,
    opacity_percentile: float = 50.0,
):
    """
    Selects a small number of readable/visible-looking base Gaussians.

    Heuristic:
      - projected inside image
      - inside central screen box
      - tile class in red/green by default
      - opacity above percentile when enough candidates exist
      - prefer closer to screen center, higher opacity, and closer depth
    """
    n = uv_all.shape[0]

    if n == 0:
        return [], {
            "candidate_count": 0,
            "final_candidate_count": 0,
            "fallback_used": "no_projection",
        }

    if target_classes is None:
        target_classes = {"red", "green"}

    cx = projector.width * 0.5
    cy = projector.height * 0.5

    center_ok = (
        (uv_all[:, 0] >= center_x_min * projector.width)
        & (uv_all[:, 0] <= center_x_max * projector.width)
        & (uv_all[:, 1] >= center_y_min * projector.height)
        & (uv_all[:, 1] <= center_y_max * projector.height)
    )

    classes = tile_classes_for_uv(
        uv=uv_all,
        valid=valid_all,
        klass_map=klass_map,
        width=projector.width,
        height=projector.height,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
    )

    class_ok = np.zeros(n, dtype=bool)
    for cls in target_classes:
        class_ok |= classes == cls

    op = np.asarray(base_opacity, dtype=np.float32).reshape(-1)
    if op.shape[0] != n:
        op = np.zeros(n, dtype=np.float32)

    opacity_norm = normalize_01(op)

    valid_depth = np.isfinite(z_all) & (z_all > 1e-6)

    # Initial strict selection
    candidate = valid_all & valid_depth & center_ok & class_ok
    fallback_used = "none"

    idx0 = np.where(candidate)[0]

    # Apply opacity threshold only if we have enough candidates.
    if idx0.size >= count * 2:
        th = float(np.percentile(op[idx0], opacity_percentile))
        candidate2 = candidate & (op >= th)

        if int(candidate2.sum()) >= count:
            candidate = candidate2
        else:
            fallback_used = "opacity_relaxed"

    # Fallback 1: allow all classes, but keep center.
    if int(candidate.sum()) < count:
        candidate = valid_all & valid_depth & center_ok
        fallback_used = "class_relaxed_center_kept"

    # Fallback 2: allow larger central region.
    if int(candidate.sum()) < count:
        larger_center_ok = (
            (uv_all[:, 0] >= 0.25 * projector.width)
            & (uv_all[:, 0] <= 0.75 * projector.width)
            & (uv_all[:, 1] >= 0.25 * projector.height)
            & (uv_all[:, 1] <= 0.75 * projector.height)
        )
        candidate = valid_all & valid_depth & larger_center_ok
        fallback_used = "larger_center_region"

    # Fallback 3: any visible projected Gaussian.
    if int(candidate.sum()) < count:
        candidate = valid_all & valid_depth
        fallback_used = "all_visible"

    idx = np.where(candidate)[0]

    if idx.size == 0:
        return [], {
            "candidate_count": 0,
            "final_candidate_count": 0,
            "fallback_used": fallback_used,
        }

    dist = np.sqrt((uv_all[idx, 0] - cx) ** 2 + (uv_all[idx, 1] - cy) ** 2)
    dist_norm = normalize_01(dist)

    depth = z_all[idx].astype(np.float32)
    depth_norm = normalize_01(depth)

    op_norm = opacity_norm[idx]

    # Lower score is better:
    # - closer to screen center
    # - closer to camera
    # - higher opacity
    score = dist_norm + 0.35 * depth_norm - 0.30 * op_norm

    order = np.argsort(score)
    chosen = idx[order[:count]]

    selected = []

    for rank, gid in enumerate(chosen, start=1):
        cls = str(classes[gid])
        selected.append({
            "id": int(gid),
            "label": f"B{rank:02d}",
            "type": "base",
            "tile_class": cls,
            "uv": [
                float(uv_all[gid, 0]),
                float(uv_all[gid, 1]),
            ],
            "depth": float(z_all[gid]),
            "opacity": float(op[gid]),
            "score": float(score[order[rank - 1]]),
        })

    info = {
        "candidate_count": int(idx0.size),
        "final_candidate_count": int(idx.size),
        "fallback_used": fallback_used,
        "target_classes": sorted(list(target_classes)),
        "center_box": {
            "x_min": center_x_min,
            "x_max": center_x_max,
            "y_min": center_y_min,
            "y_max": center_y_max,
        },
        "opacity_percentile": float(opacity_percentile),
    }

    return selected, info


def save_selected_gaussians_json(
    out_path: str,
    selected_items: list[dict],
    reference_frame: int,
    selection_info: dict,
    args,
    W: int,
    H: int,
):
    data = {
        "reference_frame": int(reference_frame),
        "description": (
            "Selected base Gaussians for visual tagging. "
            "IDs refer to indices in init_3dgs.ply/base Gaussian array."
        ),
        "selection_rule": (
            "valid projection + center-screen box + preferred tile classes "
            "+ opacity/depth/center scoring"
        ),
        "image_size": {
            "width": int(W),
            "height": int(H),
        },
        "camera": {
            "cam_pos": args.cam_pos,
            "cam_target": args.cam_target,
            "cam_up": args.cam_up,
            "cam_fov_deg": float(args.cam_fov_deg),
        },
        "tiling": {
            "tile_rows": int(args.tile_rows),
            "tile_cols": int(args.tile_cols),
            "tile_red_thresh": float(args.tile_red_thresh),
        },
        "selection_info": selection_info,
        "base": selected_items,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[DONE] selected Gaussian IDs saved to: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Build reference-view black masks for 3DGStream Gaussians using "
            "saliency-tile maps from rendered reference frames. Also optionally "
            "selects 10 visible center-screen base Gaussians for labeling."
        )
    )

    ap.add_argument("--fvv_root", required=True, help="3DGStream FVV root: contains init_3dgs.ply, NTCs/, additional_3dgs/")
    ap.add_argument("--frames_dir", required=True, help="Rendered reference frames used for saliency/tiling")
    ap.add_argument("--saliency_dir", required=True, help="Grayscale saliency maps produced from those rendered frames")
    ap.add_argument("--out_dir", required=True, help="Output folder for masks and summaries")

    ap.add_argument("--cam_pos", required=True, help="Reference camera position as x,y,z")
    ap.add_argument("--cam_target", required=True, help="Reference camera target as x,y,z")
    ap.add_argument("--cam_up", default="0,-1,0", help="Reference camera up vector as x,y,z")
    ap.add_argument("--cam_fov_deg", type=float, required=True, help="Reference camera vertical FoV in degrees")

    ap.add_argument("--tile_rows", type=int, default=DEFAULT_TILE_ROWS)
    ap.add_argument("--tile_cols", type=int, default=DEFAULT_TILE_COLS)
    ap.add_argument("--tile_red_thresh", type=float, default=DEFAULT_RED_THRESH)

    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=-1)

    ap.add_argument("--base_chunk_size", type=int, default=200000)
    ap.add_argument("--save_framewise_base_masks", action="store_true")
    ap.add_argument("--save_debug_vis", action="store_true")
    ap.add_argument("--debug_vis_dir", default="", help="Optional folder to save projected-black debug images")

    # --------------------------------------------------------
    # New selected Gaussian tagging options
    # --------------------------------------------------------
    ap.add_argument(
        "--select_visible_gaussians",
        action="store_true",
        help="Select visible center-screen base Gaussians and save selected_gaussians.json.",
    )

    ap.add_argument(
        "--selected_count",
        type=int,
        default=10,
        help="How many base Gaussians to select for labeling.",
    )

    ap.add_argument(
        "--selected_frame",
        type=int,
        default=0,
        help="Reference frame used to select the visible labeled Gaussians.",
    )

    ap.add_argument(
        "--selected_target_classes",
        default="red,green",
        help="Tile classes preferred for selection: red,green,black,any. Default: red,green.",
    )

    ap.add_argument("--center_x_min", type=float, default=0.40)
    ap.add_argument("--center_x_max", type=float, default=0.60)
    ap.add_argument("--center_y_min", type=float, default=0.35)
    ap.add_argument("--center_y_max", type=float, default=0.65)

    ap.add_argument(
        "--selected_opacity_percentile",
        type=float,
        default=50.0,
        help="Opacity percentile threshold used when enough candidates exist.",
    )

    ap.add_argument(
        "--selected_json",
        default="",
        help="Optional path for selected_gaussians.json. Default: <out_dir>/selected_gaussians.json",
    )

    ap.add_argument(
        "--save_selected_debug",
        action="store_true",
        help="Save debug image with labels B01..B10 over the selected reference frame.",
    )

    ap.add_argument(
        "--selected_debug_dir",
        default="",
        help="Optional folder for selected Gaussian label debug images. Default: <out_dir>/debug_selected",
    )

    ap.add_argument(
        "--save_selected_label_frames",
        action="store_true",
        help="Draw selected Gaussian labels on every processed frame after selection.",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    add_mask_dir = os.path.join(args.out_dir, "add_black")
    os.makedirs(add_mask_dir, exist_ok=True)

    base_frame_dir = os.path.join(args.out_dir, "base_black_framewise")
    if args.save_framewise_base_masks:
        os.makedirs(base_frame_dir, exist_ok=True)

    if args.save_debug_vis:
        dbg_dir = args.debug_vis_dir if args.debug_vis_dir else os.path.join(args.out_dir, "debug_vis")
        os.makedirs(dbg_dir, exist_ok=True)
    else:
        dbg_dir = ""

    if args.selected_debug_dir:
        selected_dbg_dir = args.selected_debug_dir
    else:
        selected_dbg_dir = os.path.join(args.out_dir, "debug_selected")

    if args.select_visible_gaussians or args.save_selected_debug or args.save_selected_label_frames:
        os.makedirs(selected_dbg_dir, exist_ok=True)

    selected_json_path = args.selected_json if args.selected_json else os.path.join(args.out_dir, "selected_gaussians.json")

    # --------------------------------------------------------
    # Pair frames and saliency maps
    # --------------------------------------------------------
    frame_names = list_images_sorted(args.frames_dir)
    sal_names = list_images_sorted(args.saliency_dir)
    sal_by_stem = {Path(x).stem: x for x in sal_names}

    pairs = []

    for f in frame_names:
        stem = Path(f).stem

        if stem in sal_by_stem:
            try:
                frame_idx = int(stem)
            except ValueError:
                continue

            pairs.append((frame_idx, f, sal_by_stem[stem]))

    pairs.sort(key=lambda t: t[0])

    if not pairs:
        raise RuntimeError("No matching frame/saliency pairs found")

    if args.end_idx < 0:
        args.end_idx = pairs[-1][0]

    pairs = [p for p in pairs if args.start_idx <= p[0] <= args.end_idx]

    if not pairs:
        raise RuntimeError("No reference frames left after applying start/end range")

    first_frame = cv2.imread(os.path.join(args.frames_dir, pairs[0][1]))
    if first_frame is None:
        raise RuntimeError("Failed to read first reference frame")

    H, W = first_frame.shape[:2]

    # --------------------------------------------------------
    # Camera/projector
    # --------------------------------------------------------
    projector = PerspectiveProjector(
        cam_pos=parse_vec3(args.cam_pos),
        cam_target=parse_vec3(args.cam_target),
        cam_up=parse_vec3(args.cam_up),
        fov_y_deg=args.cam_fov_deg,
        width=W,
        height=H,
    )

    # --------------------------------------------------------
    # Load base Gaussians and NTC/addition paths
    # --------------------------------------------------------
    base_gau = load_ply(str(Path(args.fvv_root) / "init_3dgs.ply"))
    base_xyz_np = to_numpy_xyz(base_gau.xyz)
    n_base = base_xyz_np.shape[0]

    if hasattr(base_gau, "opacity"):
        base_opacity_np = to_numpy_1d(base_gau.opacity)
    else:
        base_opacity_np = np.zeros(n_base, dtype=np.float32)

    if base_opacity_np.shape[0] != n_base:
        print(
            f"[WARN] Base opacity length mismatch: opacity={base_opacity_np.shape[0]} "
            f"base={n_base}. Opacity will not be used for selection."
        )
        base_opacity_np = np.zeros(n_base, dtype=np.float32)

    xyz_min = base_xyz_np.min(axis=0).astype(np.float32)
    xyz_max = base_xyz_np.max(axis=0).astype(np.float32)

    ntc_conf = load_ntc_config(args.fvv_root)
    ntc_paths = find_ntc_paths(args.fvv_root)
    add_paths = find_add_paths(args.fvv_root)

    if not ntc_paths:
        raise RuntimeError("No NTC_*.pth files found")

    ntc_model = build_ntc_model(ntc_conf, xyz_min, xyz_max)

    # --------------------------------------------------------
    # Mask state
    # --------------------------------------------------------
    base_black_global = np.zeros(n_base, dtype=bool)
    base_black_counts = np.zeros(n_base, dtype=np.int32)
    summary_rows = []

    selected_items = None
    selected_reference_frame = None
    selected_selection_info = None

    target_classes = parse_target_classes(args.selected_target_classes)

    # --------------------------------------------------------
    # Main loop over frames
    # --------------------------------------------------------
    for frame_idx, frame_name, sal_name in pairs:
        sal_path = os.path.join(args.saliency_dir, sal_name)
        frame_path = os.path.join(args.frames_dir, frame_name)

        sal_u8 = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)

        if sal_u8 is None:
            print(f"[WARN] Could not read saliency map for frame {frame_idx}: {sal_path}")
            continue

        _, avg_sal, klass_map = compute_tile_classes_from_saliency(
            sal_u8,
            tile_rows=args.tile_rows,
            tile_cols=args.tile_cols,
            red_thresh=args.tile_red_thresh,
        )

        ntc_state_path = ntc_paths.get(frame_idx)

        need_projection = (
            args.save_debug_vis
            or args.select_visible_gaussians
            or args.save_selected_debug
            or args.save_selected_label_frames
        )

        uv_all = None
        valid_all = None
        z_all = None

        if ntc_state_path is None:
            print(f"[WARN] Missing NTC for frame {frame_idx}; skipping base mask for this frame")
            base_black_frame = np.zeros(n_base, dtype=bool)

            if need_projection:
                uv_all = np.zeros((n_base, 2), dtype=np.float32)
                valid_all = np.zeros(n_base, dtype=bool)
                z_all = np.zeros(n_base, dtype=np.float32)

        else:
            result = compute_base_black_mask_and_projection_for_frame(
                base_xyz_np=base_xyz_np,
                ntc_model=ntc_model,
                ntc_state_path=ntc_state_path,
                projector=projector,
                klass_map=klass_map,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                chunk_size=args.base_chunk_size,
                return_projection=need_projection,
            )

            if need_projection:
                base_black_frame, uv_all, valid_all, z_all = result
            else:
                base_black_frame = result

        base_black_global |= base_black_frame
        base_black_counts += base_black_frame.astype(np.int32)

        if args.save_framewise_base_masks:
            np.save(os.path.join(base_frame_dir, f"{frame_idx:06d}.npy"), base_black_frame)

        # ----------------------------------------------------
        # Additions
        # ----------------------------------------------------
        add_path = add_paths.get(frame_idx)
        add_black_frame = np.zeros(0, dtype=bool)
        n_add = 0

        if add_path is not None:
            add_gau = load_ply(add_path)
            add_xyz_np = to_numpy_xyz(add_gau.xyz)
            n_add = add_xyz_np.shape[0]

            add_black_frame = compute_add_black_mask_for_frame(
                add_xyz_np=add_xyz_np,
                projector=projector,
                klass_map=klass_map,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
            )

        np.save(os.path.join(add_mask_dir, f"{frame_idx:06d}.npy"), add_black_frame)

        # ----------------------------------------------------
        # Existing black-projection debug visualization
        # ----------------------------------------------------
        if args.save_debug_vis:
            frame_bgr = cv2.imread(frame_path)

            if frame_bgr is not None and np.any(base_black_frame) and uv_all is not None:
                mask_all = base_black_frame & valid_all

                maybe_draw_debug_points(
                    frame_bgr=frame_bgr,
                    uv=uv_all,
                    black_mask=mask_all,
                    out_path=os.path.join(dbg_dir, f"{frame_idx:06d}_base_black.png"),
                )

        # ----------------------------------------------------
        # New visible-center selection, done once
        # ----------------------------------------------------
        if (
            args.select_visible_gaussians
            and selected_items is None
            and frame_idx >= int(args.selected_frame)
            and ntc_state_path is not None
            and uv_all is not None
            and valid_all is not None
            and z_all is not None
        ):
            selected_items, selected_selection_info = select_visible_center_base_gaussians(
                uv_all=uv_all,
                valid_all=valid_all,
                z_all=z_all,
                base_opacity=base_opacity_np,
                klass_map=klass_map,
                projector=projector,
                tile_rows=args.tile_rows,
                tile_cols=args.tile_cols,
                count=args.selected_count,
                target_classes=target_classes,
                center_x_min=args.center_x_min,
                center_x_max=args.center_x_max,
                center_y_min=args.center_y_min,
                center_y_max=args.center_y_max,
                opacity_percentile=args.selected_opacity_percentile,
            )

            selected_reference_frame = frame_idx

            save_selected_gaussians_json(
                out_path=selected_json_path,
                selected_items=selected_items,
                reference_frame=selected_reference_frame,
                selection_info=selected_selection_info,
                args=args,
                W=W,
                H=H,
            )

            print(
                f"[SELECT] frame={frame_idx:06d} selected={len(selected_items)} "
                f"fallback={selected_selection_info.get('fallback_used', '')}"
            )

            for item in selected_items:
                print(
                    f"    {item['label']} | id={item['id']} | "
                    f"class={item['tile_class']} | "
                    f"uv=({item['uv'][0]:.1f},{item['uv'][1]:.1f}) | "
                    f"depth={item['depth']:.4f} | opacity={item['opacity']:.4f}"
                )

            if args.save_selected_debug:
                frame_bgr = cv2.imread(frame_path)

                if frame_bgr is not None:
                    draw_selected_gaussian_labels(
                        frame_bgr=frame_bgr,
                        uv_all=uv_all,
                        valid_all=valid_all,
                        selected_items=selected_items,
                        out_path=os.path.join(selected_dbg_dir, f"{frame_idx:06d}_selected_gaussians.png"),
                        tile_rows=args.tile_rows,
                        tile_cols=args.tile_cols,
                        draw_grid=True,
                    )

        # ----------------------------------------------------
        # Optional: draw the selected labels on every processed frame
        # ----------------------------------------------------
        if (
            args.save_selected_label_frames
            and selected_items is not None
            and uv_all is not None
            and valid_all is not None
        ):
            frame_bgr = cv2.imread(frame_path)

            if frame_bgr is not None:
                draw_selected_gaussian_labels(
                    frame_bgr=frame_bgr,
                    uv_all=uv_all,
                    valid_all=valid_all,
                    selected_items=selected_items,
                    out_path=os.path.join(selected_dbg_dir, f"{frame_idx:06d}_labels.png"),
                    tile_rows=args.tile_rows,
                    tile_cols=args.tile_cols,
                    draw_grid=True,
                )

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------
        n_base_black_frame = int(base_black_frame.sum())
        n_add_black_frame = int(add_black_frame.sum()) if add_black_frame.size else 0

        summary_rows.append([
            frame_idx,
            frame_name,
            sal_name,
            n_base_black_frame,
            n_add,
            n_add_black_frame,
            float(avg_sal.min()),
            float(avg_sal.max()),
            float(avg_sal.mean()),
        ])

        print(
            f"[Frame {frame_idx:06d}] "
            f"base_black_this_frame={n_base_black_frame} | "
            f"base_black_global={int(base_black_global.sum())} | "
            f"add_black={n_add_black_frame}/{n_add}"
        )

    # --------------------------------------------------------
    # Save masks
    # --------------------------------------------------------
    np.save(os.path.join(args.out_dir, "base_black_global.npy"), base_black_global)
    np.save(os.path.join(args.out_dir, "base_black_counts.npy"), base_black_counts)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------
    manifest = {
        "fvv_root": os.path.abspath(args.fvv_root),
        "frames_dir": os.path.abspath(args.frames_dir),
        "saliency_dir": os.path.abspath(args.saliency_dir),
        "reference_camera": {
            "cam_pos": args.cam_pos,
            "cam_target": args.cam_target,
            "cam_up": args.cam_up,
            "cam_fov_deg": args.cam_fov_deg,
            "width": W,
            "height": H,
        },
        "tiling": {
            "tile_rows": args.tile_rows,
            "tile_cols": args.tile_cols,
            "tile_red_thresh": args.tile_red_thresh,
            "black_rule": "Gaussian center falls in a black tile of the reference frame",
        },
        "aggregation": {
            "base_union_rule": "any",
            "base_black_global_count": int(base_black_global.sum()),
            "num_reference_frames": len(summary_rows),
        },
        "selected_gaussians": {
            "enabled": bool(args.select_visible_gaussians),
            "selected_json": os.path.abspath(selected_json_path) if selected_items is not None else "",
            "selected_reference_frame": selected_reference_frame,
            "selected_count": len(selected_items) if selected_items is not None else 0,
            "selection_rule": (
                "valid projection + center-screen box + target tile classes "
                "+ opacity/depth/center scoring"
            ),
        },
    }

    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "frame_name",
            "saliency_name",
            "base_black_this_frame",
            "num_additions",
            "add_black_this_frame",
            "tile_sal_min",
            "tile_sal_max",
            "tile_sal_mean",
        ])
        writer.writerows(summary_rows)

    print(f"[DONE] base_black_global.npy saved to: {args.out_dir}")
    print(f"[DONE] add_black masks saved to: {add_mask_dir}")

    if args.save_framewise_base_masks:
        print(f"[DONE] framewise base masks saved to: {base_frame_dir}")

    if selected_items is not None:
        print(f"[DONE] selected Gaussian labels saved to: {selected_json_path}")
        print(f"[DONE] selected Gaussian debug images saved to: {selected_dbg_dir}")


if __name__ == "__main__":
    main()