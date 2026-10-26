"""
Create a tiny, fully synthetic dataset that mimics the folder layout and file
naming of the real inputs (Landsat C2 L2 ST_B10 + QA_PIXEL + MTL, Esri LULC,
Copernicus GLS NDVI, DEM/slope/aspect, Sentinel-1 stitches).

It is only meant for smoke-testing the pipeline end-to-end (does every script
run?), NOT for producing meaningful results.

Usage:
    python tests/make_synthetic_data.py --out data   (or: bash tests/smoke_test.sh)
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, rotate as nd_rotate

CRS_UTM = "EPSG:32632"
PIX = 30.0
H = W = 1400                     # Landsat-like grid size (real scenes are ~7800x7600)
X0, Y0 = 600000.0, 5400000.0     # upper-left corner (UTM 32N, southern Germany)
TRANSFORM = from_origin(X0, Y0, PIX, PIX)

QA_FILL = 1
QA_CLEAR = 21824                 # typical "clear, low confidence" value in Collection 2
QA_CLOUD = QA_CLEAR | (1 << 3) | (1 << 1)   # cloud + dilated cloud

SCENES = [
    # (sensor, acquisition date, processing date, cloud cover %, has_sar)
    ("LC08", "20190726", "20200827", 15.0, True),
    ("LC08", "20190811", "20200827", 32.0, True),
    ("LC09", "20220320", "20220322", 40.0, False),
    ("LC08", "20200614", "20200823", 60.0, True),
]


def smooth_noise(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma)
    return (f - f.min()) / (f.max() - f.min() + 1e-9)


def footprint(angle=12.0, margin=0.14):
    """Rotated rectangle of valid data, the rest is fill (like a real Landsat scene)."""
    m = np.zeros((H, W), dtype=np.float32)
    mh, mw = int(H * margin), int(W * margin)
    m[mh:H - mh, mw:W - mw] = 1
    return nd_rotate(m, angle, reshape=False, order=0) > 0.5


def write(path, arr, crs=CRS_UTM, transform=TRANSFORM, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[np.newaxis]
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                   dtype=arr.dtype, crs=crs, transform=transform, compress="lzw")
    if nodata is not None:
        profile["nodata"] = nodata
    with rio.open(path, "w", **profile) as dst:
        dst.write(arr)


def write_mtl(path, product_id, cloud_cover):
    corners = {
        "UL": (X0, Y0), "UR": (X0 + W * PIX, Y0),
        "LL": (X0, Y0 - H * PIX), "LR": (X0 + W * PIX, Y0 - H * PIX),
    }
    lines = ["GROUP = LANDSAT_METADATA_FILE", "  GROUP = PRODUCT_CONTENTS",
             f'    LANDSAT_PRODUCT_ID = "{product_id}"', "  END_GROUP = PRODUCT_CONTENTS",
             "  GROUP = IMAGE_ATTRIBUTES", f"    CLOUD_COVER = {cloud_cover:.2f}",
             f"    CLOUD_COVER_LAND = {cloud_cover:.2f}",
             "    SUN_AZIMUTH = 145.12345678", "    SUN_ELEVATION = 58.12345678",
             "  END_GROUP = IMAGE_ATTRIBUTES", "  GROUP = PROJECTION_ATTRIBUTES"]
    for k, (x, y) in corners.items():
        lines += [f"    CORNER_{k}_PROJECTION_X_PRODUCT = {x:.3f}",
                  f"    CORNER_{k}_PROJECTION_Y_PRODUCT = {y:.3f}"]
    lines += ["  END_GROUP = PROJECTION_ATTRIBUTES", "END_GROUP = LANDSAT_METADATA_FILE", "END"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(out):
    rng = np.random.default_rng(0)
    out = Path(out)
    valid = footprint()

    # --- Static layers (DEM / slope / aspect / LULC) on a slightly larger grid ---
    pad = 50
    big_t = from_origin(X0 - pad * PIX, Y0 + pad * PIX, PIX, PIX)
    bh, bw = H + 2 * pad, W + 2 * pad
    dem = (400 + 900 * smooth_noise(rng, (bh, bw), 40)).astype(np.float32)
    gy, gx = np.gradient(dem, PIX)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(-gx, gy)) % 360).astype(np.float32)
    write(out / "DEM/output_hh.tif", dem, transform=big_t, nodata=-9999.0)
    write(out / "aspect_and_slope/viz.hh_slope.tif", slope, transform=big_t)
    write(out / "aspect_and_slope/viz.hh_aspect.tif", aspect, transform=big_t)

    classes = np.array([1, 2, 4, 5, 7, 8, 9, 10, 11], dtype=np.uint8)
    lulc = classes[(smooth_noise(rng, (bh, bw), 15) * 8.999).astype(int)]
    for year in (2019, 2020, 2022):
        write(out / f"LULC_Sentinel2_10m/32U_{year}0101-{year + 1}0101.tif", lulc, transform=big_t)

    # --- NDVI (Copernicus GLS 300 m, EPSG:4326, raw DN) ---
    ndvi_t = from_origin(8.0, 49.0, 1 / 336, 1 / 336)
    for year, dates in ((2019, ["0721", "0801", "0811"]), (2020, ["0611"]), (2022, ["0321"])):
        for d in dates:
            ndvi = (100 + 120 * smooth_noise(rng, (900, 900), 10)).astype(np.uint8)
            name = f"c_gls_NDVI300_{year}{d}0000_GLOBE_PROBAV_V1.0.1__NDVI.tif"
            write(out / f"NDVI/{year}" / name, ndvi, crs="EPSG:4326", transform=ndvi_t)

    # --- Landsat scenes ---
    for sensor, acq, proc, cc, has_sar in SCENES:
        pid = f"{sensor}_L2SP_193026_{acq}_{proc}_02_T1"
        lst_k = 285 + 25 * smooth_noise(rng, (H, W), 30) + 4 * smooth_noise(rng, (H, W), 3)
        b10 = np.round((lst_k - 149.0) / 0.00341802).astype(np.uint16)
        b10[~valid] = 0

        # clouds covering roughly `cc` % of the valid area
        field = smooth_noise(rng, (H, W), 70)  # large, clumped clouds
        thr = np.quantile(field[valid], 1 - cc / 100.0)
        cloud = (field > thr) & valid
        qa = np.full((H, W), QA_FILL, dtype=np.uint16)
        qa[valid] = QA_CLEAR
        qa[cloud] = QA_CLOUD

        for folder in ("landsat_scenes", "LST_MaxCloud80"):
            write(out / folder / f"{pid}_ST_B10.TIF", b10, nodata=0)
            write(out / folder / f"{pid}_QA_PIXEL.TIF", qa, nodata=1)
        write_mtl(out / "metadata" / f"{pid}_MTL.txt", pid, cc)

        if has_sar:
            vv = (-18 + 12 * smooth_noise(rng, (H, W), 8)).astype(np.float32)
            vh = (-25 + 12 * smooth_noise(rng, (H, W), 8)).astype(np.float32)
            vv[~valid] = np.nan
            vh[~valid] = np.nan
            date_dash = f"{acq[:4]}-{acq[4:6]}-{acq[6:]}"
            name = f"S1_Stitch_{date_dash}_{sensor}_193026_{acq}.tif"
            write(out / "GEE_S1_Stitch_Exports" / name, np.stack([vv, vh]))

    # A separate "test" folder with one scene for evaluate.py
    sensor, acq, proc, *_ = SCENES[0]
    pid = f"{sensor}_L2SP_193026_{acq}_{proc}_02_T1"
    for suffix in ("ST_B10.TIF", "QA_PIXEL.TIF"):
        src = out / "landsat_scenes" / f"{pid}_{suffix}"
        dst = out / "test_data_landsat_193_26" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    print(f"Synthetic data written to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic_data")
    main(ap.parse_args().out)
