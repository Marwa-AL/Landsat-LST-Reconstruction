# Landsat LST Reconstruction

Code for the paper **"Fast Fourier Convolutional GAN for 30 m Clear-Sky Land Surface
Temperature Gap-Free Reconstruction"**
([arXiv:2607.22734](https://doi.org/10.48550/arXiv.2607.22734)).

It reconstructs cloud- and shadow-contaminated Landsat 8/9 Land Surface
Temperature (LST) at 30 m with deep generative inpainting, using auxiliary inputs: land use /
land cover, day of year, elevation, slope, aspect, NDVI, a per-overpass illumination
index and Sentinel-1 VV/VH backscatter.

Both models were developed for the paper and share the same multimodal input stack,
adversarial setup and losses; they differ only in the generator:

| Model | `generator` in config | Description |
|---|---|---|
| **Multimodal FFC-GAN** (proposed) | `ffc_lama` | LaMa generator with Fast Fourier Convolutions, adapted for LST |
| **SwinUNet-GAN** (variant) | `swin_unet` | the same multimodal GAN with the FFC generator replaced by a Swin-UNet (network architecture: Cao et al., 2022) |

- Paper: [arXiv:2607.22734](https://doi.org/10.48550/arXiv.2607.22734) (submitted to IEEE JSTARS)
- Dataset: [Marwa-AL/FFC-GAN-LST-Bavaria](https://huggingface.co/datasets/Marwa-AL/FFC-GAN-LST-Bavaria) (Hugging Face)
- Pretrained generators: see [Pretrained models](#pretrained-models)

## Repository layout

```
configs/
  data_config.yaml        # preprocessing: input/output paths, tiling, normalisation
  config.yaml             # training / evaluation / inference, Multimodal FFC-GAN
  config_swin_unet.yaml   # same, SwinUNet-GAN variant
src/
  downloading_sar_data.py      # (optional) Sentinel-1 stitches via Google Earth Engine
  generate_hillshades.py       # scene-specific hillshades from DEM + MTL sun angles
  mask_tile_generator.py       # library of real cloud masks (PNG tiles)
  generating_training_tiles.py # 21-band clear-sky training tiles
  split_dataset.py             # train / val / test split (moves files)
  custom_dataloader.py         # PyTorch Dataset with random real-mask sampling
  lama_model.py                # LaMa / FFC generator
  models.py                    # builds the generator chosen in the config
  train.py                     # GAN training
  evaluate.py                  # artificial-gap evaluation on held-out scenes
  inference.py                 # gap filling of real clouds -> GeoTIFF
tools/
  export_generator.py          # strip a training checkpoint to generator-only weights
third_party/
  Swin-Unet/                   # git submodule: Swin-UNet network code used by the SwinUNet-GAN
docker/
  Dockerfile                   # reproducible training environment
docs/
  SETUP.md                     # environment setup + how to start training
tests/
  make_synthetic_data.py       # tiny synthetic dataset with the real file naming
  smoke_test.sh                # runs the whole pipeline on it
```

## Installation

The models were trained with PyTorch 2.1.0 (CUDA 12.1, Python 3.10) inside an
enroot container based on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`.
**[docs/SETUP.md](docs/SETUP.md)** gives step-by-step instructions (enroot, Docker,
or conda), lists what must be in place before training, and covers common errors.

Short version without a container:

```bash
git clone --recursive https://github.com/Marwa-AL/Landsat-LST-Reconstruction.git
cd Landsat-LST-Reconstruction
conda create -n lst-lama python=3.10 -y && conda activate lst-lama
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`--recursive` also fetches the Swin-UNet code into `third_party/Swin-Unet` (only needed
for the SwinUNet-GAN). If you cloned without it: `git submodule update --init --recursive`.

Training downloads pretrained VGG19 (perceptual loss), LPIPS-VGG and Inception (FID)
weights on the first run, so internet access is needed once.

All commands below are run from the repository root.

## Data

### Option A: download the prepared dataset (recommended)

The training/validation/test tiles and the cloud-mask library are on Hugging Face:
**[Marwa-AL/FFC-GAN-LST-Bavaria](https://huggingface.co/datasets/Marwa-AL/FFC-GAN-LST-Bavaria)**

| File | Content | Size |
|---|---|---|
| `split_dataset_with_dem_resampledNdvi_hs_sar_vv_vh.tar` | 21-band tiles (256 × 256, float32 GeoTIFF): 29,094 train / 3,640 val / 3,638 test | ~102 GB |
| `mask_tiles.tar` | ~68,900 real cloud/shadow masks (256 × 256 PNG, 0 = clear, 255 = masked) | small |

Download and unpack into `data/` (run from the repository root):

```bash
pip install -U huggingface_hub
hf download Marwa-AL/FFC-GAN-LST-Bavaria --repo-type dataset --local-dir data
tar -xf data/split_dataset_with_dem_resampledNdvi_hs_sar_vv_vh.tar -C data
tar -xf data/mask_tiles.tar -C data
```

This gives exactly the layout `configs/config.yaml` expects:
```
data/
├── split_dataset_with_dem_resampledNdvi_hs_sar_vv_vh/{train,val,test}/tiles/*.tif
└── mask_tiles/*.png
```
You can delete the two `.tar` files after unpacking (on Windows, `tar.exe` is built in and the same commands work).

### Option B: build your own dataset

| Input | Source | Expected file names |
|---|---|---|
| Landsat 8/9 C2 L2 `ST_B10`, `QA_PIXEL`, `MTL.txt` | USGS EarthExplorer | `LC08_L2SP_193026_20190726_..._T1_ST_B10.TIF` |
| Land use / land cover (10 m) | Esri Sentinel-2 LULC | `32U_20190101-20200101.tif` |
| NDVI (300 m, 10-daily) | Copernicus Global Land Service | `c_gls_NDVI300_201907210000_..._NDVI.tif` |
| DEM, slope, aspect | e.g. Copernicus DEM GLO-30 | set in `data_config.yaml` |
| Sentinel-1 VV/VH | Google Earth Engine (`src/downloading_sar_data.py`) | `S1_Stitch_2019-07-26_LC08_193026_20190726.tif` |

Scenes without a Sentinel-1 stitch are filled with the lower normalisation bound
(normalised value 0).

```bash
python src/downloading_sar_data.py --project <gee-project> --path 193 --row 26 --start 2013-04-01 --end 2025-06-01
python src/generate_hillshades.py       --config configs/data_config.yaml
python src/mask_tile_generator.py       --config configs/data_config.yaml
python src/generating_training_tiles.py --config configs/data_config.yaml
python src/split_dataset.py             --config configs/data_config.yaml
```

### Channel layout of the training tiles

| Index | Channel |
|---|---|
| 0 | LST, normalised from [260, 342] K |
| 1–9 | one-hot LULC (Esri classes 1, 2, 4, 5, 7, 8, 9, 10, 11) |
| 10–11 | day of year (sin, cos) |
| 12 | elevation |
| 13 | slope |
| 14–15 | aspect (sin, cos) |
| 16 | NDVI |
| 17 | illumination index (per-overpass hillshade, see note below) |
| 18–19 | Sentinel-1 VV, VH |
| 20 | mask placeholder (zeros) |

**Note on channel 17.** The illumination index is computed with the cosine law of
illumination from the scene sun geometry (MTL) and DEM slope/aspect obtained with a
3 × 3 Sobel operator *without* normalisation by the pixel spacing. Slopes are therefore
strongly amplified, and the layer mainly encodes whether a pixel faces towards or away
from the sun (pixels facing away are 0). It is not a physical hillshade: a standard tool
such as `gdaldem hillshade` will not reproduce it. Use `src/generate_hillshades.py` to
build data that is compatible with the published model.

`active_data_indices` in `configs/config.yaml` selects the inputs
(`[0..11]` baseline, `[0..16]` + topography/vegetation, `[0..19]` full stack).

## Training

```bash
python src/train.py --config configs/config.yaml              # Multimodal FFC-GAN
python src/train.py --config configs/config_swin_unet.yaml    # SwinUNet-GAN
tensorboard --logdir runs
```

Checkpoints are written to `<output_dir>/checkpoints/` (`best_model.pth` = lowest
validation RMSE on the masked pixels). See [docs/SETUP.md](docs/SETUP.md) for the
full step-by-step guide.

## Pretrained models

The trained generators are attached to the
[GitHub release](https://github.com/Marwa-AL/Landsat-LST-Reconstruction/releases) of this repository:

| File | Model | Inputs |
|---|---|---|
| `ffc_gan_generator.pth` | Multimodal FFC-GAN | 20 channels + mask |
| `swin_unet_generator.pth` | SwinUNet-GAN | 20 channels + mask |

```bash
mkdir -p checkpoints
wget -P checkpoints https://github.com/Marwa-AL/Landsat-LST-Reconstruction/releases/download/v1.0/ffc_gan_generator.pth
wget -P checkpoints https://github.com/Marwa-AL/Landsat-LST-Reconstruction/releases/download/v1.0/swin_unet_generator.pth
```

`configs/config.yaml` and `configs/config_swin_unet.yaml` already point to these files.

## Evaluation and inference

```bash
# artificial gaps on held-out scenes -> metrics in <output_dir_plots>/validation_metrics.csv
python src/evaluate.py  --config configs/config.yaml
# real cloud gaps -> georeferenced GeoTIFFs
python src/inference.py --config configs/config.yaml
```
Use `configs/config_swin_unet.yaml` for the SwinUNet-GAN. To reproduce the published metrics
exactly, point `mask_lookup_csv` to the provided `validation_metrics.csv` so the same
artificial masks are used; the file used for the paper is attached to the
[v1.0 release](https://github.com/Marwa-AL/Landsat-LST-Reconstruction/releases).

## Quick check without real data

```bash
bash tests/smoke_test.sh              # Multimodal FFC-GAN
bash tests/smoke_test.sh swin_unet    # SwinUNet-GAN
```

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{alfouly2026ffcgan,
  title   = {Fast Fourier Convolutional GAN for 30 m Clear-Sky Land Surface Temperature Gap-Free Reconstruction},
  author  = {Alfouly, Marwa and Halilovic, Smajil and Bochow, Nils and Hamacher, Thomas and Boers, Niklas and Schindler, Konrad},
  journal = {arXiv preprint arXiv:2607.22734},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.22734}
}
```

## Licence

The code is released under the [MIT licence](LICENSE).
`third_party/Swin-Unet` is a submodule of
[HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet) and keeps the
terms of its authors. Data products keep their original licences (Landsat: public
domain; Copernicus products and Esri LULC: see their terms; attribution required).

## Acknowledgements

The FFC generator follows LaMa (Suvorov et al., 2022, *Resolution-robust Large Mask
Inpainting with Fourier Convolutions*). The SwinUNet-GAN uses the Swin-Unet network (Cao et al., 2022,
*Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation*).
