"""
Part of the code (CUDA and OpenGL memory transfer) is derived from:
https://github.com/jbaron34/torchwindow/tree/master
"""

from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders  # kept in case other files import it

import util
import util_gau
import numpy as np
import torch

from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.nn.functional.normalize(a)
    b_norm = torch.nn.functional.normalize(b)
    w1, x1, y1, z1 = a_norm[:, 0], a_norm[:, 1], a_norm[:, 2], a_norm[:, 3]
    w2, x2, y2, z2 = b_norm[:, 0], b_norm[:, 1], b_norm[:, 2], b_norm[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([w, x, y, z], dim=1)


@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        # sh is [N, sh_dim, 3]
        return self.sh.shape[-2]

    @torch.no_grad()
    def get_xyz_bound(self, percentile=86.6):
        half_percentile = (100 - percentile) / 200
        return (
            torch.quantile(self.xyz, half_percentile, dim=0),
            torch.quantile(self.xyz, 1 - half_percentile, dim=0),
        )

    def clone(self):
        return GaussianDataCUDA(
            xyz=self.xyz.clone(),
            rot=self.rot.clone(),
            scale=self.scale.clone(),
            opacity=self.opacity.clone(),
            sh=self.sh.clone(),
        )


@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


def gaus_cuda_from_cpu(gau) -> GaussianDataCUDA:
    """
    Converts util_gau.GaussianData (CPU) to GaussianDataCUDA (CUDA tensors).
    The expected gau fields: xyz, rot, scale, opacity, sh
    """
    gaus = GaussianDataCUDA(
        xyz=torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot=torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale=torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity=torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh=torch.tensor(gau.sh).float().cuda().requires_grad_(False),
    )
    # Ensure SH is [N, sh_dim, 3]
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus


class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()

        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1.0,
            "tanfovy": 1.0,
            "bg": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"),
            "scale_modifier": 1.0,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 1,  # updated when gaussians loaded
            "campos": None,
            "prefiltered": False,
            # Newer diff_gaussian_rasterization requires this field in settings:
            "bwd_depth": False,
            "debug": False,
        }

        gl.glViewport(0, 0, w, h)
        self.program = util.compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)

        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError("OpenGL context may be running on integrated graphics")

        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.cuda_image = None

        # FV viewer state
        self.NTCs = []
        self.additional_3dgs = []
        self.current_timestep = 0

        self.gaussians = None
        self.init_gaussians = None

        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def update_gaussian_data(self, gaus):
        self.gaussians = gaus_cuda_from_cpu(gaus)
        self.init_gaussians = GaussianDataCUDA(
            xyz=self.gaussians.xyz.clone(),
            rot=self.gaussians.rot.clone(),
            scale=self.gaussians.scale.clone(),
            opacity=self.gaussians.opacity.clone(),
            sh=self.gaussians.sh.clone(),
        )
        # sh_degree = sqrt(sh_dim) - 1
        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

    def sort_and_update(self, camera: util.Camera):
        # Viewer uses torch CUDA sorting elsewhere; keep as stub.
        pass

    def set_scale_modifier(self, modifier):
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        # Viewer may switch modes elsewhere; keep as stub.
        pass

    def set_gl_texture(self, h, w):
        # If re-allocating, try to unregister previous resource to avoid leaks.
        if getattr(self, "cuda_image", None) is not None:
            try:
                cu.cudaGraphicsUnregisterResource(self.cuda_image)
            except Exception:
                pass
            self.cuda_image = None

        if getattr(self, "tex", None) is not None:
            try:
                gl.glDeleteTextures([self.tex])
            except Exception:
                pass
            self.tex = None

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            int(w),
            int(h),
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")

    def set_render_reso(self, w, h):
        self.raster_settings["image_height"] = int(h)
        self.raster_settings["image_width"] = int(w)
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)

    @torch.no_grad()
    def query_NTC(self, xyz: torch.Tensor, timestep: int):
        if self.NTCs is None or len(self.NTCs) == 0:
            return
        if timestep < 0 or timestep >= len(self.NTCs):
            return

        mask, d_xyz, d_rot = self.NTCs[timestep](xyz)

        # make mask shape [N]
        if mask is None:
            return
        if mask.dim() > 1:
            mask = mask.squeeze(-1)
        mask = mask.bool()

        # Safety: if nothing selected, do nothing
        if mask.numel() == 0 or not torch.any(mask):
            return

        # Apply deltas ONLY where mask is true
        self.gaussians.xyz[mask] = self.gaussians.xyz[mask] + d_xyz[mask]
        self.gaussians.rot[mask] = quaternion_multiply(self.gaussians.rot[mask], d_rot[mask])


    @torch.no_grad()
    def _pad_sh_to(self, sh: torch.Tensor, target_sh_dim: int) -> torch.Tensor:
        """
        Pads SH tensor [N, sh_dim, 3] with zeros on sh_dim to reach target_sh_dim.
        """
        if sh.shape[1] == target_sh_dim:
            return sh
        if sh.shape[1] > target_sh_dim:
            # If it ever happens, truncate (safer than crashing)
            return sh[:, :target_sh_dim, :].contiguous()
        pad = torch.zeros(
            (sh.shape[0], target_sh_dim - sh.shape[1], sh.shape[2]),
            device=sh.device,
            dtype=sh.dtype,
        )
        return torch.cat([sh, pad], dim=1).contiguous()

    @torch.no_grad()
    def cat_additions(self, timestep: int) -> GaussianDataCUDA:
        """
        Concats per-frame additions with the current gaussians.
        Fixes common crash: additions SH degree != base SH degree (e.g., sh_dim 1 vs 16).
        """
        # Robust guards: prevents IndexError if additions weren't loaded.
        if self.additional_3dgs is None or len(self.additional_3dgs) == 0:
            return self.gaussians
        if timestep < 0 or timestep >= len(self.additional_3dgs):
            return self.gaussians

        additions = self.additional_3dgs[timestep]

        # Ensure SH is [N, sh_dim, 3] on both sides
        sh_add = additions.sh
        sh_base = self.gaussians.sh

        if sh_add.dim() != 3 or sh_base.dim() != 3:
            raise RuntimeError(f"Unexpected SH shapes: additions={sh_add.shape}, base={sh_base.shape}")

        if sh_add.shape[1] != sh_base.shape[1]:
            target = max(sh_add.shape[1], sh_base.shape[1])
            sh_add = self._pad_sh_to(sh_add, target)
            sh_base = self._pad_sh_to(sh_base, target)

        s2_gaussians = GaussianDataCUDA(
            xyz=torch.cat([additions.xyz, self.gaussians.xyz], dim=0),
            rot=torch.cat([additions.rot, self.gaussians.rot], dim=0),
            scale=torch.cat([additions.scale, self.gaussians.scale], dim=0),
            opacity=torch.cat([additions.opacity, self.gaussians.opacity], dim=0),
            sh=torch.cat([sh_add, sh_base], dim=0),
        )
        return s2_gaussians

    def fvv_reset(self):
        if self.init_gaussians is not None:
            self.gaussians = self.init_gaussians.clone()
        self.current_timestep = 0

    def update_camera_pose(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util.Camera):
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy

    def _rasterize(self, rendered_gaussians: GaussianDataCUDA):
        """
        diff_gaussian_rasterization has multiple API variants:
          - returns (img, radii)
          - returns (img, radii, depth, ...)
          - returns dict-like
        This adapter always returns (img, radii_or_None).
        """
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        out = rasterizer(
            means3D=rendered_gaussians.xyz,
            means2D=None,
            shs=rendered_gaussians.sh,
            colors_precomp=None,
            opacities=rendered_gaussians.opacity,
            scales=rendered_gaussians.scale,
            rotations=rendered_gaussians.rot,
            cov3D_precomp=None,
        )

        if isinstance(out, (tuple, list)):
            img = out[0]
            radii = out[1] if len(out) > 1 else None
            return img, radii

        if isinstance(out, dict):
            img = out.get("render", out.get("image", None))
            radii = out.get("radii", None)
            if img is None:
                raise RuntimeError(f"Unexpected rasterizer dict keys: {list(out.keys())}")
            return img, radii

        # Fallback for custom objects
        img = getattr(out, "render", None) or getattr(out, "image", None)
        radii = getattr(out, "radii", None)
        if img is None:
            raise RuntimeError(f"Unexpected rasterizer return type: {type(out)}")
        return img, radii

    def draw(self, timestep: int = 0):
        if self.gaussians is None:
            return  # nothing to draw yet

        # Clamp timestep to loaded NTC length (if NTCs exist), otherwise render base only.
        if self.NTCs is not None and len(self.NTCs) > 0:
            timestep = int(max(0, min(int(timestep), len(self.NTCs) - 1)))
        else:
            timestep = 0

        rendered_gaussians = self.gaussians

        with torch.no_grad():
            while timestep - self.current_timestep > 0:
                self.query_NTC(self.gaussians.xyz, self.current_timestep)
                self.current_timestep += 1

            if self.current_timestep != 0:
                rendered_gaussians = self.cat_additions(self.current_timestep - 1)

            img, _radii = self._rasterize(rendered_gaussians)

        # img expected in (C,H,W) -> convert to (H,W,C)
        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]

        # transfer CUDA tensor -> OpenGL texture via CUDA interop
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")

        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")

        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * int(width),  # float32 RGBA = 4 channels * 4 bytes
            4 * 4 * int(width),
            int(height),
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")

        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
