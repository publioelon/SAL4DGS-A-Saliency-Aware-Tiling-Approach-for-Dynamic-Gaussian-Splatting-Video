# SAL4DGS

**SAL4DGS** is a research prototype for saliency-aware tiling and Gaussian selection in dynamic Gaussian Splatting video.

The main idea is simple: instead of treating all visible Gaussians equally, we use a 2D saliency map to decide which screen-space regions are more important. Gaussians projected into low-saliency tiles can then be suppressed during rendering. This gives a practical way to study saliency-aware content reduction for dynamic 3D Gaussian Splatting scenes.

At the moment, this repository does **not** implement a full compression codec. There is no entropy coding, no new bitstream format, and no final compressed representation yet. The current version provides the saliency-aware selection layer that can later be connected to a real compression or adaptive streaming system.

The project is saliency-model agnostic. I originally tested the pipeline with ViNet, but the code does not depend on ViNet specifically. Any 2D saliency prediction model can be used, as long as it outputs one saliency map per rendered frame.

## What this repository does

SAL4DGS currently supports:

* loading dynamic 3D Gaussian Splatting scenes in a 3DGStream/FVV-style format;
* rendering dynamic Gaussian scenes with OpenGL/CUDA support;
* using external 2D saliency maps to divide the rendered view into salient and non-salient tiles;
* projecting Gaussians into the image plane and assigning them to screen-space tiles;
* generating masks for base Gaussians and per-frame added Gaussians;
* applying the generated masks inside the viewer by suppressing low-saliency Gaussians;
* sending dynamic 3DGS assets through a simple TCP sender/receiver test.

The current implementation is intended for experimentation and reproducibility, not for production use.

## Repository structure

The most important files are:

```text
main.py                 Interactive viewer. Can load a dynamic scene and apply saliency masks.
filter_gaussians.py     Generates saliency-aware Gaussian masks from rendered frames and saliency maps.
renderer.py             Offline renderer used to export frames from a dynamic Gaussian scene.
renderer_cuda.py        CUDA-based renderer backend.
renderer_ogl.py         OpenGL renderer backend.
live_tcp.py             TCP receiver and live loading state for dynamic 3DGS assets.
tcp_fvv_sender.py       TCP sender for init_3dgs, NTC files, and per-frame Gaussian additions.
util.py                 Camera and OpenGL utility functions.
util_gau.py             Gaussian PLY loading utilities.
util_3dgstream.py       Helpers for loading NTCs and additional Gaussians.
NTC.py                  Neural Transformation Cache support.
guess_ntc_config.py     Helper script for recovering a missing NTC config.json in some cases.
```

The code is kept as a flat Python folder because several scripts use local imports.

## Expected dynamic scene format

The viewer expects a dynamic Gaussian Splatting scene with this structure:

```text
scene_root/
    init_3dgs.ply

    NTCs/
        config.json
        NTC_000000.pth
        NTC_000001.pth
        NTC_000002.pth
        ...

    additional_3dgs/
        additions_000000.ply
        additions_000001.ply
        additions_000002.ply
        ...
```

For example, if the scene has:

```text
NTC_000000.pth ... NTC_000298.pth
additions_000000.ply ... additions_000298.ply
```

then the viewer should usually be run with:

```text
--frames 300
```

and the TCP sender should be run with:

```text
--start 0 --end 298
```

The reason is that the initial Gaussian set is stored in `init_3dgs.ply`, while the NTC and additions files describe the following dynamic frames.

## Saliency map format

SAL4DGS does not run a saliency model by itself. It expects saliency maps to be generated beforehand by any external 2D saliency predictor.

The rendered frames and the saliency maps should have matching filenames:

```text
rendered_frames/
    000000.png
    000001.png
    000002.png
    ...

saliency_maps/
    000000.png
    000001.png
    000002.png
    ...
```

The saliency maps should be grayscale images. Brighter pixels are treated as more salient.

A saliency model can be ViNet, UNISAL, DeepGaze, TranSalNet, U2Net, or any other model that produces framewise saliency maps. The only requirement is that the output images match the rendered frames.

## Installation

This project was tested in a Windows/Python environment with CUDA support. Some dependencies, especially CUDA rendering and tiny-cuda-nn, may require manual setup depending on your machine.

A typical setup is:

```powershell
git clone https://github.com/publioelon/SAL4DGS-A-Saliency-Aware-Tiling-Approach-for-Dynamic-Gaussian-Splatting-Video.git
cd SAL4DGS-A-Saliency-Aware-Tiling-Approach-for-Dynamic-Gaussian-Splatting-Video
```

Create and activate a Python environment. For example:

```powershell
conda create -n sal4dgs python=3.8
conda activate sal4dgs
```

Then install the basic Python dependencies:

```powershell
pip install -r requirements.txt
```

Depending on your setup, you may also need to install or compile the following packages manually:

```text
PyTorch with CUDA
tiny-cuda-nn
diff-gaussian-rasterization
PyOpenGL
glfw
imgui
plyfile
opencv-python
numpy
imageio
```

CUDA-related packages are the most likely part to require manual adjustment. If the CUDA renderer fails, first check whether PyTorch detects the GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

## Running the viewer without saliency masks

After preparing a dynamic scene, run:

```powershell
python main.py ^
  --autoload_fvv "C:\path\to\scene_root" ^
  --frames 300 ^
  --autoplay ^
  --video_fps 30
```

This loads the dynamic Gaussian scene and plays it in the viewer without applying any saliency-aware mask.

## Running the viewer with saliency masks

If you already have generated masks, run:

```powershell
python main.py ^
  --autoload_fvv "C:\path\to\scene_root" ^
  --black_mask_dir "C:\path\to\scene_root\black_masks" ^
  --frames 300 ^
  --autoplay ^
  --video_fps 30
```

If you used a different output folder, replace `black_masks` with that folder. For example:

```powershell
python main.py ^
  --autoload_fvv "C:\path\to\scene_root" ^
  --black_mask_dir "C:\path\to\scene_root\black_masks_vinet_e" ^
  --frames 300 ^
  --autoplay ^
  --video_fps 30
```

The mask folder is expected to contain files such as:

```text
black_masks/
    base_black_global.npy
    base_black_counts.npy
    manifest.json
    summary.csv

    add_black/
        000000.npy
        000001.npy
        ...

    base_black_framewise/
        000000.npy
        000001.npy
        ...
```

## Rendering frames for saliency prediction

To generate saliency maps, you first need rendered reference frames from the dynamic scene.

Example:

```powershell
python renderer.py ^
  --fvv_root "C:\path\to\scene_root" ^
  --out_dir "C:\path\to\rendered_frames" ^
  --frames 300 ^
  --width 1280 ^
  --height 720 ^
  --camera_mode auto_bbox ^
  --cam_fov_deg 75
```

This exports rendered images that can then be passed to any 2D saliency prediction model.

## Generating saliency-aware Gaussian masks

Once you have rendered frames and matching saliency maps, run:

```powershell
python filter_gaussians.py ^
  --fvv_root "C:\path\to\scene_root" ^
  --frames_dir "C:\path\to\rendered_frames" ^
  --saliency_dir "C:\path\to\saliency_maps" ^
  --out_dir "C:\path\to\output_black_masks" ^
  --tile_rows 5 ^
  --tile_cols 9 ^
  --tile_red_thresh 0.22 ^
  --save_framewise_base_masks ^
  --save_debug_vis
```

The tiling parameters define how the image is divided. In the example above, the frame is divided into 5 rows and 9 columns.

The saliency threshold controls which tiles are considered highly salient. Tiles above the threshold are treated as important, neighboring tiles can be preserved as a safety region, and the remaining low-saliency tiles are marked for suppression.

After the masks are generated, they can be loaded by the viewer using:

```powershell
python main.py ^
  --autoload_fvv "C:\path\to\scene_root" ^
  --black_mask_dir "C:\path\to\output_black_masks" ^
  --frames 300 ^
  --autoplay ^
  --video_fps 30
```

## Live TCP test

The repository also includes a simple TCP sender/receiver test. This is useful for experimenting with live delivery of dynamic 3DGS/FVV assets.

Start the receiver first:

```powershell
python main.py ^
  --tcp_listen 6009 ^
  --tcp_bind 127.0.0.1 ^
  --tcp_cache "C:\tmp\gs_stream_cache\live_session" ^
  --tcp_clear_cache ^
  --frames 300 ^
  --autoplay ^
  --video_fps 30
```

Then, in another terminal, start the sender:

```powershell
python tcp_fvv_sender.py ^
  --host 127.0.0.1 ^
  --port 6009 ^
  --root "C:\path\to\scene_root" ^
  --start 0 ^
  --end 298 ^
  --poll_ms 50
```

Adjust `--end` according to the last available NTC/additions index in your scene.

## Typical workflow

A full experiment usually follows this order:

```text
1. Prepare a dynamic Gaussian Splatting scene.
2. Render reference frames with renderer.py.
3. Run any 2D saliency prediction model on those frames.
4. Save one saliency map per rendered frame.
5. Run filter_gaussians.py to generate Gaussian masks.
6. Open the scene in main.py with --black_mask_dir.
7. Compare rendering with and without the saliency-aware masks.
```

This makes the saliency model replaceable. The Gaussian filtering stage does not care which saliency model produced the maps.

## Notes on compression

This repository should not be interpreted as a complete compression method yet.

The current implementation performs saliency-aware Gaussian selection and suppression. This can be used to study which Gaussians are less relevant from a perceptual/saliency perspective, and it can serve as a control layer for later compression.

A future version could connect this selection layer to:

```text
bitrate control
entropy coding
3DGS pruning
NTC compression
network-adaptive streaming
region-of-interest encoding
quality-level assignment per tile
```

For now, the output is a set of masks and a viewer that can apply them.

## Troubleshooting

If the viewer cannot import a module, first check that you are running the command from the repository root. The current code uses local imports.

If the viewer cannot find `init_3dgs.ply`, check that `--autoload_fvv` points to the scene root, not to the `NTCs` folder.

If the TCP sender appears to hang, check whether the final frame index exists. For example, if your last file is `NTC_000298.pth`, use:

```powershell
--end 298
```

Do not use `--end 299` unless `NTC_000299.pth` actually exists.

If the saliency mask generation fails, check that the rendered frame names and saliency map names match exactly.

If CUDA/OpenGL interop fails, make sure the application is running on the NVIDIA GPU and not on an integrated GPU. This is especially important on laptops.

## Project status

SAL4DGS is still an experimental research codebase. Some scripts are practical tools developed during experiments, so the code is not yet packaged as a polished Python library.

The goal of this first release is reproducibility: users should be able to inspect the pipeline, run the viewer, generate masks from their own saliency maps, and reproduce the saliency-aware Gaussian suppression behavior.

## Citation

If this repository is useful for your research, please cite the repository for now. A formal paper citation will be added if this work is published.

## License

Please check the license file included in this repository. Some parts of the code build on ideas and dependencies from existing Gaussian Splatting and 3DGStream-style projects, so users should also respect the licenses of the upstream projects and external dependencies.
