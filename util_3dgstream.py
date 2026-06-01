import os
import json
import glob
from typing import Optional, Dict, Any, List

import numpy as np
import tinycudann as tcnn
import torch

from plyfile import PlyData, PlyElement
from NTC import NeuralTransformationCache
from renderer_cuda import GaussianDataCUDA, gaus_cuda_from_cpu
from util_gau import load_ply


@torch.no_grad()
def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 1e-6, 1.0 - 1e-6)
    return torch.log(x / (1 - x))


def construct_list_of_attributes(gau_cuda: GaussianDataCUDA) -> List[str]:
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # DC (3)
    for i in range(3):
        l.append(f"f_dc_{i}")
    # Rest of SH
    for i in range((gau_cuda.sh_dim - 1) * 3):
        l.append(f"f_rest_{i}")
    l.append("opacity")
    for i in range(gau_cuda.scale.shape[1]):
        l.append(f"scale_{i}")
    for i in range(gau_cuda.rot.shape[1]):
        l.append(f"rot_{i}")
    return l


def _normpath(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(p)))


def _find_config_json(fvv_path: str) -> Optional[str]:
    """
    Expected: <FVV_path>/NTCs/config.json
    Also checks a few nearby fallback locations.
    """
    fvv_path = _normpath(fvv_path)
    candidates = [
        os.path.join(fvv_path, "NTCs", "config.json"),
        os.path.join(fvv_path, "config.json"),
        os.path.join(os.path.dirname(fvv_path), "NTCs", "config.json"),
        os.path.join(os.path.dirname(fvv_path), "config.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _try_extract_config_from_ckpt(ckpt_obj: Any) -> Optional[Dict[str, Any]]:
    """
    If the trainer embedded encoding/network config in the checkpoint dict,
    recover it here. (Only works if training saved it.)
    """
    if isinstance(ckpt_obj, dict):
        # common patterns
        for key in ("config", "cfg", "NTC_conf", "ntc_conf"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                d = ckpt_obj[key]
                if "encoding" in d and "network" in d:
                    return d

        # sometimes stored flat
        if "encoding" in ckpt_obj and "network" in ckpt_obj:
            if isinstance(ckpt_obj["encoding"], dict) and isinstance(ckpt_obj["network"], dict):
                return {"encoding": ckpt_obj["encoding"], "network": ckpt_obj["network"]}

    return None


def load_NTCs(FVV_path: str, gau_cuda: GaussianDataCUDA, total_frames: Optional[int] = None):
    """
    Loads NTC_*.pth and builds NeuralTransformationCache list.

    total_frames:
      - if None: auto = number_of_ntc_files + 1
      - else: loads min(total_frames-1, found_files)
    """
    fvv_path = _normpath(FVV_path)
    ntc_dir = os.path.join(fvv_path, "NTCs")

    ntc_paths = sorted(glob.glob(os.path.join(ntc_dir, "NTC_*.pth")))
    if len(ntc_paths) == 0:
        raise FileNotFoundError(
            f"No NTC_*.pth found in: {ntc_dir}\n"
            "Expected files like NTC_000000.pth, NTC_000001.pth, ..."
        )

    if total_frames is None:
        total_frames = len(ntc_paths) + 1

    # keep only what we need (total_frames-1 deltas)
    ntc_paths = ntc_paths[: max(0, total_frames - 1)]

    # xyz bounds used by NeuralTransformationCache
    xyz_min, xyz_max = gau_cuda.get_xyz_bound()

    # config.json or checkpoint-embedded config
    config_path = _find_config_json(fvv_path)
    ntc_conf: Optional[Dict[str, Any]] = None

    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            ntc_conf = json.load(f)

    if ntc_conf is None:
        first = torch.load(ntc_paths[0], map_location="cpu")
        ntc_conf = _try_extract_config_from_ckpt(first)

    if ntc_conf is None:
        raise FileNotFoundError(
            "Could not find NTC config.\n\n"
            "The viewer needs the tinycudann configs (encoding + network), usually at:\n"
            f"  {os.path.join(fvv_path, 'NTCs', 'config.json')}\n\n"
            "But that file is missing, and your NTC_*.pth files do not appear to embed it.\n"
            "Fix: export/copy the config JSON that was used in training (ntc_conf_path) into that location."
        )

    if ("encoding" not in ntc_conf) or ("network" not in ntc_conf):
        raise ValueError(
            f"Invalid NTC config (missing 'encoding'/'network'). Source: {config_path or 'checkpoint'}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntcs: List[NeuralTransformationCache] = []
    for _ in ntc_paths:
        model = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=8,
            encoding_config=ntc_conf["encoding"],
            network_config=ntc_conf["network"],
        ).to(device)
        model.eval()
        ntcs.append(NeuralTransformationCache(model, xyz_min, xyz_max))

    # load weights
    for i, ntc in enumerate(ntcs):
        ckpt = torch.load(ntc_paths[i], map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        ntc.load_state_dict(state, strict=False)

    return ntcs


def load_Additions(FVV_path: str, total_frames: Optional[int] = None):
    """
    Loads additions_*.ply if present. Returns list[GaussianDataCUDA].
    If folder doesn't exist, returns [] (viewer can still run).
    """
    fvv_path = _normpath(FVV_path)
    add_dir = os.path.join(fvv_path, "additional_3dgs")

    if not os.path.isdir(add_dir):
        return []

    add_paths = sorted(glob.glob(os.path.join(add_dir, "additions_*.ply")))
    if len(add_paths) == 0:
        return []

    if total_frames is None:
        total_frames = len(add_paths) + 1

    add_paths = add_paths[: max(0, total_frames - 1)]
    additions_gaus = [load_ply(p) for p in add_paths]
    additions_gaus_cuda = [gaus_cuda_from_cpu(g) for g in additions_gaus]
    return additions_gaus_cuda


def get_per_frame_3dgs(FVV_path, gau_cuda: GaussianDataCUDA, total_frames: int = 150):
    raise NotImplementedError("This function is not implemented yet")


def save_gau_cuda(gau_cuda: GaussianDataCUDA, path: str):
    xyz = gau_cuda.xyz.detach().cpu().numpy()
    rotation = gau_cuda.rot.detach().cpu().numpy()
    normals = np.zeros_like(xyz)

    f_dc = (
        gau_cuda.sh[:, 0:1, :]
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )
    f_rest = (
        gau_cuda.sh[:, 1:, :]
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .detach()
        .cpu()
        .numpy()
    )

    opacities = inverse_sigmoid(gau_cuda.opacity).detach().cpu().numpy()
    scale = torch.log(torch.clamp(gau_cuda.scale, min=1e-12)).detach().cpu().numpy()

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(gau_cuda)]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
