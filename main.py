import argparse
import math
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog

import glfw
import imageio
import imgui
import numpy as np
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer

import util
import util_3dgstream
import util_gau
from live_tcp import LiveTCPState, infer_total_frames
from renderer_ogl import OpenGLRenderer

try:
    import torch
except Exception:
    torch = None


dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

g_camera = util.Camera(720, 1280)

BACKEND_OGL = 0
BACKEND_CUDA = 1

g_renderer_list = [None]
g_renderer_idx = BACKEND_OGL
g_renderer = g_renderer_list[g_renderer_idx]

g_scale_modifier = 1.0
g_auto_sort = False

g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False

g_render_mode_tables = [
    "Gaussian Ball",
    "Flat Ball",
    "Billboard",
    "Depth",
    "SH:0",
    "SH:0~1",
    "SH:0~2",
    "SH:0~3 (default)",
]
g_render_mode = 7

g_FVV_path = ""

VIDEO_FPS = 30.0
VIDEO_INTERVAL = 1.0 / VIDEO_FPS
g_last_frame_time = 0.0
g_timestep = 0
g_paused = True
g_reset = False
g_total_frame = 300

# -----------------------------
# Foveated overlay controls
# -----------------------------
g_show_fovea_overlay = True
g_show_fovea_fill = True
g_fovea_use_center = True
g_fovea_center_x = 0.5
g_fovea_center_y = 0.5
g_fovea_inner_deg = 30.0
g_fovea_outer_deg = 45.0
g_fovea_segments = 128
g_fovea_thickness = 2.0
g_fovea_dark_alpha = 0.45

# -----------------------------
# Black-mask state
# -----------------------------
g_black_mask_dir = ""
g_black_mask_loaded = False
g_black_mask_status = "disabled"
g_black_mask_error = ""
g_black_mask_warned = set()
g_base_global_mask = None
g_current_base_mask_kind = "none"
g_current_base_mask_count = 0
g_current_add_mask_count = 0
g_base_orig_opacity = None
g_add_orig_opacity = {}


def impl_glfw_init():
    window_name = "Tiny 3DGStream Viewer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    global window
    window = glfw.create_window(g_camera.w, g_camera.h, window_name, None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(0)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window


def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False

    g_camera.process_mouse(xpos, ypos)


def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return

    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)


def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)


def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)


def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False


def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False


def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)


def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)


def autoload_session_if_requested():
    global g_FVV_path, g_total_frame, g_paused, g_timestep, g_last_frame_time
    global g_renderer

    if not args.autoload_fvv:
        return

    fvv = args.autoload_fvv
    ply_path = os.path.join(fvv, "init_3dgs.ply")
    if not os.path.exists(ply_path):
        print(f"[AUTOLOAD] init_3dgs.ply not found at: {ply_path}")
        return

    gaussians = util_gau.load_ply(ply_path)
    g_renderer.update_gaussian_data(gaussians)
    g_renderer.sort_and_update(g_camera)

    if args.frames > 0:
        g_total_frame = args.frames
    else:
        g_total_frame = infer_total_frames(fvv)

    g_FVV_path = fvv
    g_renderer.NTCs = util_3dgstream.load_NTCs(
        g_FVV_path, g_renderer.gaussians, g_total_frame
    )
    g_renderer.additional_3dgs = util_3dgstream.load_Additions(
        g_FVV_path, g_total_frame
    )

    g_timestep = 0
    g_last_frame_time = time.time()
    g_paused = not args.autoplay
    reset_mask_runtime_state()


def angular_radius_to_pixels(theta_deg: float, fovy_rad: float, height_px: int) -> float:
    theta_deg = max(0.001, float(theta_deg))
    theta_deg = min(theta_deg, 89.0)
    theta = math.radians(theta_deg)
    denom = math.tan(max(1e-6, fovy_rad * 0.5))
    r_px = (math.tan(theta) / denom) * (height_px * 0.5)
    return max(0.0, float(r_px))


def get_fovea_center_pixels():
    if g_fovea_use_center:
        return g_camera.w * 0.5, g_camera.h * 0.5

    cx = np.clip(g_fovea_center_x, 0.0, 1.0) * g_camera.w
    cy = np.clip(g_fovea_center_y, 0.0, 1.0) * g_camera.h
    return float(cx), float(cy)


def draw_foveated_overlay():
    global g_fovea_outer_deg

    if not g_show_fovea_overlay:
        return

    if g_fovea_outer_deg < g_fovea_inner_deg:
        g_fovea_outer_deg = g_fovea_inner_deg

    cx, cy = get_fovea_center_pixels()
    r_inner = angular_radius_to_pixels(g_fovea_inner_deg, g_camera.fovy, g_camera.h)
    r_outer = angular_radius_to_pixels(g_fovea_outer_deg, g_camera.fovy, g_camera.h)

    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(g_camera.w, g_camera.h)

    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_INPUTS
        | imgui.WINDOW_NO_BACKGROUND
        | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
        | imgui.WINDOW_NO_NAV
    )

    imgui.begin("##fovea_overlay", False, flags)
    draw_list = imgui.get_window_draw_list()

    col_inner = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)
    col_outer = imgui.get_color_u32_rgba(1.0, 0.85, 0.15, 1.0)
    col_center = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0)

    if g_show_fovea_fill:
        col_dark = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, g_fovea_dark_alpha)
        draw_list.add_rect_filled(0, 0, g_camera.w, max(0.0, cy - r_outer), col_dark)
        draw_list.add_rect_filled(
            0, min(float(g_camera.h), cy + r_outer), g_camera.w, g_camera.h, col_dark
        )

        y0 = max(0.0, cy - r_outer)
        y1 = min(float(g_camera.h), cy + r_outer)
        draw_list.add_rect_filled(0, y0, max(0.0, cx - r_outer), y1, col_dark)
        draw_list.add_rect_filled(
            min(float(g_camera.w), cx + r_outer), y0, g_camera.w, y1, col_dark
        )

    draw_list.add_circle(
        cx, cy, r_outer, col_outer, g_fovea_segments, g_fovea_thickness
    )
    draw_list.add_circle(
        cx, cy, r_inner, col_inner, g_fovea_segments, g_fovea_thickness
    )

    cross_half = 6.0
    draw_list.add_line(cx - cross_half, cy, cx + cross_half, cy, col_center, 1.5)
    draw_list.add_line(cx, cy - cross_half, cx, cy + cross_half, col_center, 1.5)

    imgui.end()


def fmt_vec3(v):
    return f"{v[0]:.6f},{v[1]:.6f},{v[2]:.6f}"


def print_camera_state_to_console():
    pos = fmt_vec3(g_camera.position)
    target = fmt_vec3(g_camera.target)
    up = fmt_vec3(g_camera.up)
    fov_deg = np.rad2deg(g_camera.fovy)

    print("\n[CAMERA STATE]")
    print(f"position = ({pos})")
    print(f"target = ({target})")
    print(f"up = ({up})")
    print(f"yaw = {g_camera.yaw:.6f}")
    print(f"pitch = {g_camera.pitch:.6f}")
    print(f"fov_deg = {fov_deg:.6f}")
    print(f"target_dist = {g_camera.target_dist:.6f}")
    print("[renderer.py args]")
    print(
        f"--cam_pos={pos} --cam_target={target} --cam_up={up} --cam_fov_deg={fov_deg:.6f}\n"
    )


# ------------------------------------------------------------
# Mask helpers
# ------------------------------------------------------------
def reset_mask_runtime_state():
    global g_base_orig_opacity, g_add_orig_opacity
    global g_current_base_mask_kind, g_current_base_mask_count, g_current_add_mask_count

    g_base_orig_opacity = None
    g_add_orig_opacity = {}
    g_current_base_mask_kind = "none"
    g_current_base_mask_count = 0
    g_current_add_mask_count = 0


def _warn_once(key, msg):
    if key in g_black_mask_warned:
        return
    g_black_mask_warned.add(key)
    print(msg)


def _is_torch_tensor(x):
    return (torch is not None) and isinstance(x, torch.Tensor)


def _clone_array(x):
    if _is_torch_tensor(x):
        return x.clone()
    return np.array(x, copy=True)


def _restore_array(dst, src):
    if _is_torch_tensor(dst):
        dst.copy_(src)
    else:
        dst[...] = src


def _set_zero_by_mask(arr, mask_bool):
    if _is_torch_tensor(arr):
        mask_t = torch.as_tensor(mask_bool, device=arr.device, dtype=torch.bool)
        arr[mask_t] = 0
    else:
        arr[mask_bool] = 0


def _load_npy_if_exists(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except Exception as e:
        print(f"[MASK] Failed to load {path}: {repr(e)}")
        return None


def _base_global_mask_candidates(mask_dir):
    return [
        os.path.join(mask_dir, "base_black_global.npy"),
        os.path.join(mask_dir, "base_black.npy"),
    ]


def _base_counts_mask_path(mask_dir):
    return os.path.join(mask_dir, "base_black_counts.npy")


def _find_framewise_base_mask_path(mask_dir, idx):
    base_dir = os.path.join(mask_dir, "base_black_framewise")
    if not os.path.isdir(base_dir):
        return None

    candidates = [
        os.path.join(base_dir, f"{idx:06d}.npy"),
        os.path.join(base_dir, f"base_black_{idx:06d}.npy"),
        os.path.join(base_dir, f"{idx:04d}.npy"),
        os.path.join(base_dir, f"base_black_{idx:04d}.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _find_add_mask_path(mask_dir, idx):
    add_dir = os.path.join(mask_dir, "add_black")
    if not os.path.isdir(add_dir):
        return None

    candidates = [
        os.path.join(add_dir, f"{idx:06d}.npy"),
        os.path.join(add_dir, f"add_black_{idx:06d}.npy"),
        os.path.join(add_dir, f"{idx:04d}.npy"),
        os.path.join(add_dir, f"add_black_{idx:04d}.npy"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_mask_dir_once():
    global g_black_mask_dir, g_black_mask_loaded, g_black_mask_status, g_black_mask_error
    global g_base_global_mask

    if not args.black_mask_dir:
        g_black_mask_status = "disabled"
        return

    if g_black_mask_loaded:
        return

    g_black_mask_dir = os.path.normpath(
        os.path.abspath(os.path.expanduser(args.black_mask_dir))
    )

    try:
        counts = _load_npy_if_exists(_base_counts_mask_path(g_black_mask_dir))
        if counts is not None and args.base_black_min_count > 1:
            g_base_global_mask = counts >= int(args.base_black_min_count)
            g_black_mask_status = f"global from counts >= {args.base_black_min_count}"
        else:
            g_base_global_mask = None
            for c in _base_global_mask_candidates(g_black_mask_dir):
                arr = _load_npy_if_exists(c)
                if arr is not None:
                    g_base_global_mask = arr.astype(bool)
                    g_black_mask_status = f"global from {os.path.basename(c)}"
                    break

        if g_base_global_mask is None:
            if os.path.isdir(os.path.join(g_black_mask_dir, "base_black_framewise")):
                g_black_mask_status = "framewise base available"
            elif os.path.isdir(os.path.join(g_black_mask_dir, "add_black")):
                g_black_mask_status = "additions only"
            else:
                g_black_mask_status = "no masks found"

        g_black_mask_loaded = True
        g_black_mask_error = ""

        print(f"[MASK] dir = {g_black_mask_dir}")
        print(f"[MASK] status = {g_black_mask_status}")
        if g_base_global_mask is not None:
            print(
                f"[MASK] global base loaded: {int(g_base_global_mask.sum())}/{len(g_base_global_mask)}"
            )

    except Exception as e:
        g_black_mask_error = repr(e)
        g_black_mask_status = "error"
        g_black_mask_loaded = True
        print(f"[MASK] ERROR: {g_black_mask_error}")


def _ensure_base_opacity_snapshot():
    global g_base_orig_opacity

    gaus = getattr(g_renderer, "gaussians", None)
    if gaus is None or not hasattr(gaus, "opacity"):
        return False

    if g_base_orig_opacity is None:
        g_base_orig_opacity = _clone_array(gaus.opacity)
        return True

    try:
        if gaus.opacity.shape != g_base_orig_opacity.shape:
            g_base_orig_opacity = _clone_array(gaus.opacity)
    except Exception:
        g_base_orig_opacity = _clone_array(gaus.opacity)

    return True


def _ensure_add_opacity_snapshot(idx):
    adds = getattr(g_renderer, "additional_3dgs", None)
    if adds is None:
        return None

    try:
        add_obj = adds[idx]
    except Exception:
        return None

    if add_obj is None or not hasattr(add_obj, "opacity"):
        return None

    snap = g_add_orig_opacity.get(idx, None)
    if snap is None:
        g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)
    else:
        try:
            if add_obj.opacity.shape != snap.shape:
                g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)
        except Exception:
            g_add_orig_opacity[idx] = _clone_array(add_obj.opacity)

    return add_obj


def restore_original_opacities_for_current_frame():
    if _ensure_base_opacity_snapshot():
        gaus = getattr(g_renderer, "gaussians", None)
        _restore_array(gaus.opacity, g_base_orig_opacity)

    add_obj = _ensure_add_opacity_snapshot(g_timestep)
    if add_obj is not None:
        _restore_array(add_obj.opacity, g_add_orig_opacity[g_timestep])


def _get_base_mask_for_frame(idx, n_gaussians):
    framewise_path = _find_framewise_base_mask_path(g_black_mask_dir, idx)
    if framewise_path is not None:
        m = _load_npy_if_exists(framewise_path)
        if m is not None:
            m = m.astype(bool)
            if len(m) == n_gaussians:
                return m, "framewise"
            _warn_once(
                ("frame_base_len", idx, n_gaussians),
                f"[MASK] framewise base mask length mismatch at frame {idx}: mask={len(m)} gaussians={n_gaussians}",
            )

    if g_base_global_mask is not None:
        if len(g_base_global_mask) == n_gaussians:
            return g_base_global_mask, "global"
        _warn_once(
            ("global_base_len", n_gaussians),
            f"[MASK] global base mask length mismatch: mask={len(g_base_global_mask)} gaussians={n_gaussians}",
        )

    return None, "none"


def apply_opacity_masks_for_current_frame():
    global g_current_base_mask_kind, g_current_base_mask_count, g_current_add_mask_count
    global g_black_mask_error

    g_current_base_mask_kind = "none"
    g_current_base_mask_count = 0
    g_current_add_mask_count = 0

    if not args.black_mask_dir:
        return

    load_mask_dir_once()

    gaus = getattr(g_renderer, "gaussians", None)
    if gaus is None or not hasattr(gaus, "opacity"):
        return

    restore_original_opacities_for_current_frame()

    try:
        n_base = int(gaus.opacity.shape[0])
    except Exception:
        return

    base_mask, kind = _get_base_mask_for_frame(g_timestep, n_base)
    g_current_base_mask_kind = kind

    if base_mask is not None:
        try:
            _set_zero_by_mask(gaus.opacity, base_mask)
            g_current_base_mask_count = int(base_mask.sum())
        except Exception as e:
            g_black_mask_error = repr(e)
            _warn_once(
                ("base_apply", g_timestep, repr(e)),
                f"[MASK] base apply error at frame {g_timestep}: {repr(e)}",
            )

    add_obj = _ensure_add_opacity_snapshot(g_timestep)
    add_path = _find_add_mask_path(g_black_mask_dir, g_timestep)

    if add_obj is not None and add_path is not None:
        add_mask = _load_npy_if_exists(add_path)
        if add_mask is not None:
            add_mask = add_mask.astype(bool)
            try:
                n_add = int(add_obj.opacity.shape[0])
            except Exception:
                n_add = -1

            if n_add >= 0 and len(add_mask) == n_add:
                try:
                    _set_zero_by_mask(add_obj.opacity, add_mask)
                    g_current_add_mask_count = int(add_mask.sum())
                except Exception as e:
                    g_black_mask_error = repr(e)
                    _warn_once(
                        ("add_apply", g_timestep, repr(e)),
                        f"[MASK] additions apply error at frame {g_timestep}: {repr(e)}",
                    )
            else:
                _warn_once(
                    ("add_len", g_timestep, n_add),
                    f"[MASK] additions mask length mismatch at frame {g_timestep}: mask={len(add_mask)} additions={n_add}",
                )


def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort
    global g_show_control_win, g_show_help_win, g_show_camera_win
    global g_render_mode, g_render_mode_tables
    global g_FVV_path, g_paused, g_reset, g_timestep, g_last_frame_time, g_total_frame, VIDEO_FPS, VIDEO_INTERVAL
    global g_show_fovea_overlay, g_show_fovea_fill, g_fovea_use_center
    global g_fovea_center_x, g_fovea_center_y, g_fovea_inner_deg, g_fovea_outer_deg, g_fovea_dark_alpha

    VIDEO_FPS = float(args.video_fps)
    VIDEO_INTERVAL = 1.0 / max(1e-6, VIDEO_FPS)

    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5

    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    root = tk.Tk()
    root.withdraw()

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_window_size_callback(window, window_resize_callback)

    g_renderer_list[BACKEND_OGL] = OpenGLRenderer(g_camera.w, g_camera.h)
    try:
        from renderer_cuda import CUDARenderer
        g_renderer_list += [CUDARenderer(g_camera.w, g_camera.h)]
    except ImportError:
        pass

    if len(g_renderer_list) > 1:
        g_renderer_idx = BACKEND_CUDA
    else:
        g_renderer_idx = BACKEND_OGL

    g_renderer = g_renderer_list[g_renderer_idx]
    gaussians = util_gau.naive_gaussian()
    update_activated_renderer_state(gaussians)

    g_last_frame_time = time.time()
    load_mask_dir_once()

    live = None
    live_autoplay_started = False

    if args.tcp_listen is not None:
        g_FVV_path = args.tcp_cache
        g_total_frame = args.frames if args.frames > 0 else 300
        live = LiveTCPState(cache_root=args.tcp_cache, total_frames_hint=g_total_frame, autoplay=args.autoplay)
        live.start_receiver(
            bind_host=args.tcp_bind,
            port=args.tcp_listen,
            clear_cache=args.tcp_clear_cache,
        )

    if live is None:
        autoload_session_if_requested()

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()

        if live is not None:
            live.process_new_files(g_renderer, g_camera)
            if live.autoplay and (not live_autoplay_started):
                if live.init_ply_ok and live.config_ok and live.max_play >= 0:
                    g_paused = False
                    g_timestep = 0
                    g_last_frame_time = time.time()
                    live_autoplay_started = True
                    reset_mask_runtime_state()

        max_play = (g_total_frame - 1)
        if live is not None and live.max_play >= 0:
            max_play = live.max_play

        current_time = time.time()
        if (current_time - g_last_frame_time) >= VIDEO_INTERVAL and not g_paused:
            if g_timestep < max_play:
                g_timestep += 1
            g_last_frame_time = current_time

        if g_reset:
            g_renderer.fvv_reset()
            g_reset = False
            g_last_frame_time = time.time()
            reset_mask_runtime_state()

        if g_timestep > max_play:
            g_timestep = max_play if max_play >= 0 else 0

        apply_opacity_masks_for_current_frame()

        try:
            g_renderer.draw(g_timestep)
        except Exception as e:
            if live is not None:
                with live._lock:
                    live.err = f"draw error: {repr(e)}"
            else:
                print("[DRAW] ERROR:", repr(e))

        draw_foveated_overlay()

        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                _, g_show_control_win = imgui.menu_item("Show Control", None, g_show_control_win)
                _, g_show_help_win = imgui.menu_item("Show Help", None, g_show_help_win)
                _, g_show_camera_win = imgui.menu_item("Show Camera Control", None, g_show_camera_win)
                imgui.end_menu()
            imgui.end_main_menu_bar()

        if g_show_control_win:
            if imgui.begin("Control", True):
                changed, g_renderer_idx = imgui.combo(
                    "backend", g_renderer_idx, ["ogl", "cuda"][: len(g_renderer_list)]
                )
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)
                    if live is not None:
                        g_renderer.NTCs = live.ntc_list
                        g_renderer.additional_3dgs = live.add_list
                    reset_mask_runtime_state()

                imgui.text(f"# of Gaus = {len(getattr(g_renderer, 'gaussians', gaussians))}")
                imgui.text(f"Render FPS = {imgui.get_io().framerate:.1f}")
                imgui.text(f"Video FPS = {VIDEO_FPS:.1f}")
                imgui.text(f"FVV Dir: {g_FVV_path}")
                imgui.text(f"Frame {g_timestep}")

                if args.black_mask_dir:
                    imgui.separator()
                    imgui.text("Opacity masks:")
                    imgui.text(f" dir: {g_black_mask_dir}")
                    imgui.text(f" status: {g_black_mask_status}")
                    imgui.text(f" base mode: {g_current_base_mask_kind}")
                    imgui.text(f" base selected: {g_current_base_mask_count}")
                    imgui.text(f" add selected: {g_current_add_mask_count}")
                    if g_black_mask_error:
                        imgui.text(f" err: {g_black_mask_error}")

                if live is not None:
                    snap = live.ui_snapshot()
                    imgui.separator()
                    imgui.text("TCP LIVE:")
                    imgui.text(f" cache: {snap['cache']}")
                    imgui.text(f" status: {snap['status']} ({snap['connected']})")
                    if snap["err"]:
                        imgui.text(f" err: {snap['err']}")
                    imgui.text(f" init_ply: {snap['init_ply']}")
                    imgui.text(f" config: {snap['config']}")
                    imgui.text(f" ready_ntc: {snap['ready_ntc']}")
                    imgui.text(f" ready_add: {snap['ready_add']}")
                    imgui.text(f" max_play: {snap['max_play']}")
                    imgui.text(f" files: {snap['files']} ({snap['gb']:.2f} GB)")

                imgui.text("#Frames:")
                imgui.same_line()
                max_frames = max(300, g_total_frame)
                _, g_total_frame = imgui.slider_int("frames", g_total_frame, 1, max_frames)

                if imgui.button("Pause"):
                    g_paused = True
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Play"):
                    g_paused = False
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Reset"):
                    g_paused = True
                    g_reset = True
                    g_timestep = 0
                    g_last_frame_time = time.time()

                imgui.same_line()
                if imgui.button("Step"):
                    g_timestep += 1
                    g_last_frame_time = time.time()

                if imgui.button(label="load ply"):
                    file_path = filedialog.askopenfilename(
                        title="load ply",
                        initialdir="C:\\Users",
                        filetypes=[("ply file", ".ply")],
                    )
                    if file_path:
                        try:
                            gaussians = util_gau.load_ply(file_path)
                            g_renderer.update_gaussian_data(gaussians)
                            g_renderer.sort_and_update(g_camera)
                            reset_mask_runtime_state()
                        except RuntimeError:
                            pass

                imgui.same_line()
                if imgui.button(label="save ply"):
                    file_path = filedialog.asksaveasfilename(
                        title="save ply",
                        initialdir="C:\\Users\\",
                        defaultextension=".ply",
                        filetypes=[("ply file", ".ply")],
                    )
                    if file_path:
                        try:
                            util_3dgstream.save_gau_cuda(g_renderer.gaussians, file_path)
                        except RuntimeError:
                            pass

                imgui.same_line()
                if imgui.button(label="load FVV"):
                    dirp = filedialog.askdirectory(title="load FVV", initialdir="C:\\Users")
                    if dirp:
                        try:
                            g_FVV_path = dirp
                            g_renderer.NTCs = util_3dgstream.load_NTCs(
                                g_FVV_path, g_renderer.gaussians, g_total_frame
                            )
                            g_renderer.additional_3dgs = util_3dgstream.load_Additions(
                                g_FVV_path, g_total_frame
                            )
                            reset_mask_runtime_state()
                        except RuntimeError:
                            pass

                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()

                changed, g_scale_modifier = imgui.slider_float(
                    "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                )
                imgui.same_line()
                if imgui.button(label="reset"):
                    g_scale_modifier = 1.0
                    changed = True
                if changed:
                    g_renderer.set_scale_modifier(g_scale_modifier)

                changed, g_render_mode = imgui.combo(
                    "shading", g_render_mode, g_render_mode_tables
                )
                if changed:
                    g_renderer.set_render_mod(g_render_mode - 4)

                if imgui.button(label="sort Gaussians"):
                    g_renderer.sort_and_update(g_camera)
                imgui.same_line()
                changed, g_auto_sort = imgui.checkbox("auto sort", g_auto_sort)
                if g_auto_sort:
                    g_renderer.sort_and_update(g_camera)

                if imgui.button(label="save image"):
                    width, height = glfw.get_framebuffer_size(window)
                    gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                    gl.glReadBuffer(gl.GL_FRONT)
                    bufferdata = gl.glReadPixels(
                        0,
                        0,
                        width,
                        height,
                        gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE,
                    )
                    img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                    imageio.imwrite("save.png", img[::-1])

                imgui.separator()
                imgui.text("Foveated overlay")
                _, g_show_fovea_overlay = imgui.checkbox("show overlay", g_show_fovea_overlay)
                _, g_show_fovea_fill = imgui.checkbox(
                    "darken outside outer circle", g_show_fovea_fill
                )
                _, g_fovea_use_center = imgui.checkbox(
                    "use viewport center", g_fovea_use_center
                )
                _, g_fovea_inner_deg = imgui.slider_float(
                    "inner deg",
                    g_fovea_inner_deg,
                    1.0,
                    60.0,
                    "inner = %.1f deg",
                )
                _, g_fovea_outer_deg = imgui.slider_float(
                    "outer deg",
                    g_fovea_outer_deg,
                    1.0,
                    80.0,
                    "outer = %.1f deg",
                )
                _, g_fovea_dark_alpha = imgui.slider_float(
                    "dark alpha",
                    g_fovea_dark_alpha,
                    0.0,
                    1.0,
                    "alpha = %.2f",
                )

                if not g_fovea_use_center:
                    _, g_fovea_center_x = imgui.slider_float(
                        "center x", g_fovea_center_x, 0.0, 1.0, "x = %.3f"
                    )
                    _, g_fovea_center_y = imgui.slider_float(
                        "center y", g_fovea_center_y, 0.0, 1.0, "y = %.3f"
                    )

                cx, cy = get_fovea_center_pixels()
                r_inner = angular_radius_to_pixels(g_fovea_inner_deg, g_camera.fovy, g_camera.h)
                r_outer = angular_radius_to_pixels(g_fovea_outer_deg, g_camera.fovy, g_camera.h)

                imgui.text(f"center px = ({cx:.1f}, {cy:.1f})")
                imgui.text(f"inner px = {r_inner:.1f}")
                imgui.text(f"outer px = {r_outer:.1f}")

                imgui.end()

        if g_show_camera_win:
            imgui.begin("Camera Control", True)
            imgui.text("Camera state")
            imgui.separator()
            imgui.text(
                f"pos = ({g_camera.position[0]:.6f}, {g_camera.position[1]:.6f}, {g_camera.position[2]:.6f})"
            )
            imgui.text(
                f"target = ({g_camera.target[0]:.6f}, {g_camera.target[1]:.6f}, {g_camera.target[2]:.6f})"
            )
            imgui.text(
                f"up = ({g_camera.up[0]:.6f}, {g_camera.up[1]:.6f}, {g_camera.up[2]:.6f})"
            )
            imgui.text(f"yaw = {g_camera.yaw:.6f}")
            imgui.text(f"pitch = {g_camera.pitch:.6f}")
            imgui.text(f"fov_deg = {np.rad2deg(g_camera.fovy):.6f}")
            imgui.text(f"target_dist = {g_camera.target_dist:.6f}")
            imgui.separator()
            imgui.text("renderer.py args")
            imgui.text(f"--cam_pos={fmt_vec3(g_camera.position)}")
            imgui.text(f"--cam_target={fmt_vec3(g_camera.target)}")
            imgui.text(f"--cam_up={fmt_vec3(g_camera.up)}")
            imgui.text(f"--cam_fov_deg={np.rad2deg(g_camera.fovy):.6f}")

            if imgui.button("Print camera to console"):
                print_camera_state_to_console()

            imgui.separator()
            if imgui.button(label="rot 180"):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                "t", g_camera.target_dist, 1.0, 8.0, "target dist = %.3f"
            )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
            )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

            imgui.end()

        if g_show_help_win:
            imgui.begin("Help", True)
            imgui.text("Open Gaussian Splatting PLY file by clicking 'load ply'")
            imgui.text("Use left click & move to rotate camera")
            imgui.text("Use right click & move to translate camera")
            imgui.text("Press Q/E to roll camera")
            imgui.text("Use scroll to zoom in/out")
            imgui.text("Use control panel to change setting")
            imgui.text("Foveated overlay: inner circle = focus region")
            imgui.text("Outer circle = reduced search region")
            imgui.text("Darkening is applied outside the outer circle")
            imgui.text("Open 'Window -> Show Camera Control' to see live camera values")
            imgui.text("Use 'Print camera to console' to get renderer.py arguments")
            imgui.text(
                "Use --black_mask_dir to apply framewise/global base masks and add masks with opacity=0"
            )
            imgui.end()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    if live is not None:
        live.request_stop()

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny 3DGStream Viewer.")
    parser.add_argument("--hidpi", action="store_true")
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--tcp_listen", type=int, default=None)
    parser.add_argument("--tcp_bind", default="127.0.0.1")
    parser.add_argument("--tcp_cache", default=r"C:\tmp\gs_stream_cache\live_session")
    parser.add_argument("--tcp_clear_cache", action="store_true")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--autoload_fvv", default="")
    parser.add_argument(
        "--black_mask_dir",
        default="",
        help="Folder containing base_black_framewise/, base_black_global.npy and add_black/",
    )
    parser.add_argument(
        "--base_black_min_count",
        type=int,
        default=1,
        help="If >1 and base_black_counts.npy exists, use counts >= this threshold for global fallback",
    )
    args = parser.parse_args()
    main()