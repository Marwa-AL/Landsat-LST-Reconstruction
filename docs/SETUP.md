# Setup and training guide

This guide reproduces the environment the published model was trained in and
walks through everything needed to start a training run.

## 1. What was used

| Component | Version used for the paper |
|---|---|
| GPU | NVIDIA RTX 6000 Ada (48 GB); any CUDA GPU with ≥ 16 GB should work with `batch_size: 16` |
| Host OS / driver | Ubuntu Linux, NVIDIA driver ≥ 530 (CUDA 12.1 capable) |
| Container runtime | [enroot](https://github.com/NVIDIA/enroot) 3.5 (Docker works too) |
| Base image | `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (Ubuntu 20.04, Python 3.10) |
| PyTorch | 2.1.0, CUDA 12.1, cuDNN 8.9 |
| Other packages | exact versions in [`requirements.txt`](../requirements.txt) |

Get the code including the Swin-UNet submodule (only needed for the SwinUNet-GAN):
```bash
git clone --recursive https://github.com/Marwa-AL/Landsat-LST-Reconstruction.git
```

Pick **one** of the options in section 2. Option A is what was used for the paper.

## 2. Create the environment

### Option A: enroot (recommended, no root rights needed)

```bash
# A1. Download the base image once (creates a ~3.5 GB .sqsh file in the current folder)
enroot import docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# A2. Create a persistent container from it (unpacked to ~/.local/share/enroot)
enroot create --name lst-lama pytorch+pytorch+2.1.0-cuda12.1-cudnn8-runtime.sqsh

# A3. Start it WRITABLE (--rw) with the repository mounted at /workspace
enroot start --rw --mount /path/to/this/repo:/workspace lst-lama

# A4. Inside the container: install the project packages once
cd /workspace
pip install -r requirements.txt
```

Packages installed in A4 stay in the `lst-lama` container. From now on, only A3
is needed to start working.

Three rules that avoid most problems:
- Always start the **named container** (`lst-lama`), not the `.sqsh` file. Starting
  `enroot start ... image.sqsh` directly gives a throw-away container in which
  everything you install is lost on exit.
- Always use `--rw`, otherwise `pip install` cannot write into the container.
- If the data lives outside the repository, add a second mount, e.g.
  `--mount /path/to/data:/workspace/data`.

Optional: save the finished environment as one file (to copy it to another machine):
```bash
enroot export --output lst-lama.sqsh lst-lama          # run outside the container
enroot create --name lst-lama lst-lama.sqsh            # on the other machine
```

### Option B: Docker

```bash
docker build -t lst-lama -f docker/Dockerfile .
docker run --gpus all --shm-size=16g -it -v "$PWD":/workspace -w /workspace lst-lama
```

### Option C: without a container (conda / venv)

```bash
conda create -n lst-lama python=3.10 -y && conda activate lst-lama
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Check the environment

Inside the container / environment:
```bash
python -c "import torch, torchmetrics, lpips, rasterio, seaborn, tensorboard; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available(), '| packages OK')"
```
Expected: `torch 2.1.0 | GPU: True | packages OK`.

## 3. What must be in place before training

```
/workspace/                         <- repository root (mounted)
├── configs/config.yaml             <- training settings and data paths
├── src/train.py
└── data/
    ├── split_dataset_with_dem_resampledNdvi_hs_sar_vv_vh/   <- split_data_dir
    │   ├── train/tiles/*.tif       <- 21-band tiles (see README, "Channel layout")
    │   ├── val/tiles/*.tif
    │   └── test/tiles/*.tif
    └── mask_tiles/*.png            <- mask_dir: real cloud masks (0 / 255)
```

Get `data/` either from the Hugging Face dataset
([Marwa-AL/FFC-GAN-LST-Bavaria](https://huggingface.co/datasets/Marwa-AL/FFC-GAN-LST-Bavaria),
download and unpack commands in the README) or by running the preprocessing scripts
(README, section "Data"). Then check these lines in `configs/config.yaml`:

| Key | Meaning |
|---|---|
| `split_data_dir` | folder that contains `train/tiles`, `val/tiles`, `test/tiles` |
| `mask_dir` | folder with the PNG mask tiles |
| `active_data_indices` | input channels; `[0..19]` = full multi-modal model |
| `run_name`, `output_dir` | TensorBoard run name and checkpoint folder (use a new name per experiment) |
| `resume_checkpoint` | `null` for a new run, or the path of an `epoch_XX.pth` to continue |
| `batch_size`, `num_epochs` | 16 and 60 for the paper; lower `batch_size` if the GPU runs out of memory |
| `preview_tiles` | file names inside `val/tiles` to plot after each epoch (missing ones are skipped) |

Quick check that the paths are right (inside the container):
```bash
ls data/split_dataset_with_dem_resampledNdvi_hs_sar_vv_vh/train/tiles | head -3
ls data/mask_tiles | head -3
```

## 4. Start training

Long runs should be started inside `tmux`, so they continue when the terminal or
SSH connection closes:

```bash
tmux new -s lst                                             # on the host
enroot start --rw --mount /path/to/this/repo:/workspace lst-lama
cd /workspace
export MPLCONFIGDIR=/tmp/mpl                                # avoids a Matplotlib cache warning
export TORCH_HOME=/workspace/.cache/torch                   # keep downloaded weights between runs
python src/train.py --config configs/config.yaml              # FFC-GAN
# or: python src/train.py --config configs/config_swin_unet.yaml   (SwinUNet-GAN)
```

Detach with `Ctrl+b` then `d`; re-attach with `tmux attach -t lst`.

What you should see:
1. `Using device: cuda`, the number of training / validation tiles found, and the
   generator / discriminator architectures.
2. On the very first run: downloads of the VGG19 (perceptual loss), LPIPS and
   Inception (FID) weights (~1.2 GB, needs internet once).
3. A progress bar per epoch, then validation metrics (RMSE on the masked pixels,
   MAE, SSIM, LPIPS, FID).

Outputs:
- `<output_dir>/checkpoints/epoch_XX.pth` every epoch and `best_model.pth` (lowest validation RMSE)
- `<output_dir>/validation_previews/*.png`
- TensorBoard logs in `runs/<run_name>`: `tensorboard --logdir runs`

To **resume** an interrupted run, set `resume_checkpoint` to the last
`<output_dir>/checkpoints/epoch_XX.pth` and start the same command again. The
script prints `Resuming training from checkpoint ...`. If it prints
`Checkpoint not found ... Starting from scratch`, the path is wrong.

## 5. Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torchmetrics'` (or `tensorboard`, `lpips`) | Started the `.sqsh` image directly or without `--rw`, so installed packages were lost | Start the named container with `--rw` (section 2, A3) and run `pip install -r requirements.txt` once |
| `nvml error: driver/library version mismatch` / `98-nvidia.sh exited with return code 1` | The host NVIDIA driver was updated but the old one is still loaded | Reboot the host, or reload the `nvidia` kernel modules. To prevent it: `sudo apt-mark hold` the nvidia packages |
| `No training tiles found in ...` | `split_data_dir` does not match where the data is mounted | Compare the path in the message with `ls` inside the container; fix the mount or the config |
| `Checkpoint not found ... Starting from scratch` | Wrong `resume_checkpoint` | Point it to an existing `epoch_XX.pth`, or set it to `null` for a new run |
| `CUDA out of memory` | GPU too small for `batch_size: 16` | Lower `batch_size` (e.g. 8) |
| `bus error` / DataLoader shared-memory errors (Docker) | Docker's default 64 MB `/dev/shm` | Add `--shm-size=16g`, or set `num_workers: 0` |
| Matplotlib "created a temporary cache directory" / pip "cache has been disabled" | Home folder not writable inside the container | Harmless; `export MPLCONFIGDIR=/tmp/mpl` silences the first |
| Training stops when the terminal closes | Not running inside `tmux` | Start it in `tmux` (section 4) |
