
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import glfw
import imageio.v2 as imageio
import numpy as np
import OpenGL.GL as gl

# Make local imports work when this file sits next to the 3DGStream code.
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
if DIR_PATH not in sys.path:
    sys.path.append(DIR_PATH)
os.chdir(DIR_PATH)

import util
import util_gau
import util_3dgstream
from renderer_ogl import OpenGLRenderer

try:
    from renderer_cuda import CUDARenderer
except Exception:
    CUDARenderer = None


def infer_total_frames(fvv_root: str) -> int:
    ntc_dir = Path(fvv_root) / "NTCs"
    ntcs = sorted(ntc_dir.glob("NTC_*.pth"))
    return max(1, len(ntcs) + 1)


def parse_vec3(s: str) -> np.ndarray:
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected vec3 'x,y,z', got: {s}")
    return np.array(vals, dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def compute_scene_center_and_extent(gaussians_cpu: util_gau.GaussianData) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    xyz = gaussians_cpu.xyz.astype(np.float32)
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    center = (xyz_min + xyz_max) * 0.5
    extent = xyz_max - xyz_min
    radius = float(max(np.max(extent), 1e-3))
    return center, xyz_min, xyz_max, radius


def make_camera(
    width: int,
    height: int,
    gaussians_cpu: util_gau.GaussianData,
    args: argparse.Namespace,
) -> util.Camera:
    cam = util.Camera(height, width)
    center, xyz_min, xyz_max, radius = compute_scene_center_and_extent(gaussians_cpu)

    # Highest priority: fully manual camera.
    if args.cam_pos and args.cam_target:
        cam.position = parse_vec3(args.cam_pos)
        cam.target = parse_vec3(args.cam_target)
    else:
        mode = args.camera_mode.lower()
        up = parse_vec3(args.cam_up)

        if mode == "auto_bbox":
            # Place the camera outside the scene, looking at the center.
            # Default axis is +Z, but can be changed.
            view_axis = normalize(parse_vec3(args.view_axis))
            cam.position = center + view_axis * (args.cam_distance_mult * radius)
            cam.target = center.copy()
        elif mode == "center_forward":
            # Place the camera near the center of the scene, looking forward.
            forward = normalize(parse_vec3(args.forward_dir))
            cam.position = center.copy() + parse_vec3(args.center_offset)
            cam.target = cam.position + forward * max(radius * 0.5, 1.0)
        else:
            raise ValueError(f"Unknown camera mode: {args.camera_mode}")

        cam.up = up

    cam.fovy = np.deg2rad(float(args.cam_fov_deg))
    cam.is_pose_dirty = True
    cam.is_intrin_dirty = True
    return cam


class HiddenGLContext:
    def __init__(self, width: int, height: int, show_window: bool = False):
        self.width = width
        self.height = height
        self.show_window = show_window
        self.window = None

    def __enter__(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self.show_window else glfw.FALSE)

        self.window = glfw.create_window(self.width, self.height, "renderer", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create OpenGL window/context")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)
        gl.glViewport(0, 0, self.width, self.height)
        return self.window

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
        finally:
            glfw.terminate()


def capture_backbuffer_rgb(width: int, height: int) -> np.ndarray:
    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
    gl.glReadBuffer(gl.GL_BACK)
    bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = np.frombuffer(bufferdata, np.uint8).reshape(height, width, 3)
    return img[::-1].copy()  # vertical flip to top-left origin


def build_renderer(width: int, height: int):
    # Match main.py: prefer CUDA backend when available.
    if CUDARenderer is not None:
        print("[INFO] Using CUDARenderer backend")
        return CUDARenderer(width, height)
    print("[WARN] CUDARenderer not available; falling back to OpenGLRenderer")
    return OpenGLRenderer(width, height)


def load_dynamic_scene(renderer, camera: util.Camera, fvv_root: str, total_frames: int):
    ply_path = os.path.join(fvv_root, "init_3dgs.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"init_3dgs.ply not found: {ply_path}")

    gaussians_cpu = util_gau.load_ply(ply_path)
    renderer.update_gaussian_data(gaussians_cpu)
    renderer.sort_and_update(camera)
    renderer.set_scale_modifier(1.0)
    renderer.set_render_mod(4)  # same startup path used by main.py
    renderer.update_camera_pose(camera)
    renderer.update_camera_intrin(camera)
    renderer.set_render_reso(camera.w, camera.h)

    if not hasattr(renderer.gaussians, "get_xyz_bound"):
        raise RuntimeError(
            "Dynamic FVV export requires the CUDA renderer backend. "
            "renderer.gaussians has no get_xyz_bound(); make sure renderer_cuda.py is available "
            "and CUDA works in this environment."
        )

    # load_NTCs / load_Additions expect renderer.gaussians as in main.py
    renderer.NTCs = util_3dgstream.load_NTCs(fvv_root, renderer.gaussians, total_frames)
    renderer.additional_3dgs = util_3dgstream.load_Additions(fvv_root, total_frames)

    return gaussians_cpu


def export_frames(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = args.frames if args.frames > 0 else infer_total_frames(args.fvv_root)
    width = int(args.width)
    height = int(args.height)

    # Need the CPU Gaussian data once to place the camera well.
    init_ply = os.path.join(args.fvv_root, "init_3dgs.ply")
    gaussians_cpu_for_camera = util_gau.load_ply(init_ply)
    camera = make_camera(width, height, gaussians_cpu_for_camera, args)

    with HiddenGLContext(width, height, show_window=args.show_window):
        renderer = build_renderer(width, height)
        gaussians_cpu = load_dynamic_scene(renderer, camera, args.fvv_root, total_frames)

        # If camera_mode relies on scene stats, recompute after load is fine too.
        # Keep manual settings if provided.
        if not (args.cam_pos and args.cam_target):
            camera = make_camera(width, height, gaussians_cpu, args)
            renderer.update_camera_pose(camera)
            renderer.update_camera_intrin(camera)
            renderer.set_render_reso(camera.w, camera.h)
            renderer.sort_and_update(camera)

        print("[INFO] Export settings")
        print(f"  FVV root : {args.fvv_root}")
        print(f"  Out dir  : {out_dir}")
        print(f"  Frames   : {total_frames}")
        print(f"  Size     : {width}x{height}")
        print(f"  Cam pos  : {camera.position}")
        print(f"  Cam tgt  : {camera.target}")
        print(f"  Cam up   : {camera.up}")
        print(f"  Cam fov  : {np.rad2deg(camera.fovy):.2f} deg")

        if hasattr(renderer, "fvv_reset"):
            try:
                renderer.fvv_reset()
            except Exception:
                pass

        t_start = time.time()
        saved = 0

        for t in range(total_frames):
            gl.glViewport(0, 0, width, height)
            gl.glClearColor(0.0, 0.0, 0.0, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

            if not args.no_sort_every_frame:
                renderer.sort_and_update(camera)

            renderer.draw(t)
            gl.glFinish()

            frame_rgb = capture_backbuffer_rgb(width, height)
            out_path = out_dir / f"{t:06d}.png"
            imageio.imwrite(out_path, frame_rgb)
            saved += 1

            if (t + 1) % max(1, args.log_every) == 0 or (t + 1) == total_frames:
                elapsed = time.time() - t_start
                fps = saved / max(elapsed, 1e-6)
                print(f"[EXPORT] {t + 1}/{total_frames} -> {out_path.name}  ({fps:.2f} frames/s)")

        elapsed = time.time() - t_start
        print(f"[DONE] Exported {saved} frames in {elapsed:.2f}s ({saved / max(elapsed, 1e-6):.2f} frames/s)")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Offline frame exporter for a 3DGStream FVV session")
    ap.add_argument("--fvv_root", required=True, help="Path containing init_3dgs.ply, NTCs/, additional_3dgs/")
    ap.add_argument("--out_dir", required=True, help="Output directory for PNG frames")
    ap.add_argument("--frames", type=int, default=0, help="Number of frames to export. 0 = infer from NTC count")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--show_window", action="store_true", help="Show the render window while exporting")
    ap.add_argument("--no_sort_every_frame", action="store_true",
                    help="Disable re-sorting Gaussians before each frame")
    ap.add_argument("--log_every", type=int, default=10)

    # Camera control
    ap.add_argument("--camera_mode", default="auto_bbox", choices=["auto_bbox", "center_forward"],
                    help="auto_bbox: outside scene looking at center. center_forward: observer near center looking along forward_dir")
    ap.add_argument("--cam_pos", default="", help="Manual camera position x,y,z")
    ap.add_argument("--cam_target", default="", help="Manual camera target x,y,z")
    ap.add_argument("--cam_up", default="0,-1,0", help="Camera up vector x,y,z")
    ap.add_argument("--cam_fov_deg", type=float, default=75.0, help="Vertical field of view in degrees")

    # Auto camera parameters
    ap.add_argument("--view_axis", default="0,0,1",
                    help="For auto_bbox: direction from center to place the camera, x,y,z")
    ap.add_argument("--cam_distance_mult", type=float, default=2.5,
                    help="For auto_bbox: distance multiplier relative to scene radius")
    ap.add_argument("--forward_dir", default="0,0,-1",
                    help="For center_forward: forward viewing direction x,y,z")
    ap.add_argument("--center_offset", default="0,0,0",
                    help="For center_forward: offset added to the scene center x,y,z")

    return ap


def main():
    args = build_argparser().parse_args()
    export_frames(args)


if __name__ == "__main__":
    main()
