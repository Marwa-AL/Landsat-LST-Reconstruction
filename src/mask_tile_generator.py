"""
Builds the library of realistic cloud/shadow mask tiles (PNG, 0/255) from the
Landsat QA_PIXEL bands. These masks are sampled randomly during training.

Usage (from the repository root):
    python src/mask_tile_generator.py --config configs/data_config.yaml
"""
import argparse
import gc
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import yaml
from PIL import Image
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
from skimage.transform import rotate


def find_landsat_files(folder, suffix):
    """Case-insensitive search (Landsat files end in .TIF; Linux globbing is case-sensitive)."""
    suffix = suffix.upper()
    return sorted(p for p in Path(folder).iterdir() if p.name.upper().endswith(suffix))

# --- Helper Functions (copied from the main script for consistency) ---

def get_raster_bounds(raster_path):
    """Extracts the geographical bounds, CRS, transform, height, and width."""
    with rio.open(raster_path) as src:
        return src.bounds, src.crs, src.transform, src.height, src.width

def convert_b10toKelvin(array):
    """Converts scaled ST_B10 pixel values to Kelvin."""
    return array * 0.00341802 + 149.0
    
def qa_masks(qa):
    """Parses Landsat Collection 2 QA_PIXEL bits and returns a dictionary of masks."""
    qa = np.asarray(qa, dtype=np.uint32)
    cloud = (qa & (1 << 3)) != 0
    cloud_shad = (qa & (1 << 4)) != 0
    cirrus = (qa & (1 << 2)) != 0
    dilated = (qa & (1 << 1)) != 0
    snow = (qa & (1 << 5)) != 0
    return {'cloud': cloud, 'shadow': cloud_shad, 'cirrus': cirrus, 'dilated': dilated, 'snow': snow}

def polygon_rotation_angle(polygon):
    """Calculates the rotation angle of a polygon's minimum rotated rectangle."""
    if polygon.is_empty: return 0
    coords = list(polygon.minimum_rotated_rectangle.exterior.coords)
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    return np.degrees(np.arctan2(dy, dx)) % 180

def crop_and_unrotate(array, polygon, angle_deg, transform):
    """
    User's original function.
    Works for 2D (H,W) or 3D (C,H,W) arrays.
    Crops to bounding box of valid (non-NaN) data, then rotates back.
    """
    if array.ndim == 3:
        valid_2d = ~np.isnan(array).all(axis=0)
    else:
        valid_2d = ~np.isnan(array)

    if not np.any(valid_2d):
        return np.array([]), transform

    rows, cols = np.where(valid_2d)
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    cropped = array[..., min_row:max_row+1, min_col:max_col+1]

    if cropped.ndim == 3:
        unrotated = np.stack([
            rotate(c, angle=90-angle_deg, resize=False, order=0, mode='constant', cval=np.nan)
            for c in cropped
        ])
    else:
        unrotated = rotate(cropped, angle=90-angle_deg, resize=False, order=0, mode='constant', cval=np.nan)

    if unrotated.ndim == 3:
        valid_mask = ~np.isnan(unrotated).all(axis=0)
    else:
        valid_mask = ~np.isnan(unrotated)

    props = regionprops(label(valid_mask))
    if props:
        minr, minc, maxr, maxc = props[0].bbox
        unrotated_cropped = unrotated[..., minr:maxr, minc:maxc]
    else:
        unrotated_cropped = unrotated

    return unrotated_cropped, transform

def trim_nan_edges_limited(array, border=128):
    """
    User's original function.
    Trims up to `border` rows/cols from edges if they contain NaNs.
    """
    if array.size == 0: return array, 0, 0
    if array.ndim == 3:
        nan_mask = np.any(np.isnan(array), axis=0)
    else:
        nan_mask = np.isnan(array)

    nrows, ncols = nan_mask.shape
    top, bottom, left, right = 0, nrows, 0, ncols

    for i in range(border):
        if np.any(nan_mask[top, :]): top += 1
        else: break
    for i in range(border):
        if np.any(nan_mask[bottom-1, :]): bottom -= 1
        else: break
    for i in range(border):
        if np.any(nan_mask[:, left]): left += 1
        else: break
    for i in range(border):
        if np.any(nan_mask[:, right-1]): right -= 1
        else: break
    
    trimmed_array = array[..., top:bottom, left:right]
    return trimmed_array, top, left

# --- New Verification Plotting Function ---
def plot_mask_comparison(original_lst, final_mask, date_str, output_dir):
    """
    Saves a side-by-side plot of the original LST and the final processed mask.
    """
    output_dir.mkdir(exist_ok=True)
    
    # Convert LST to Kelvin for better visualization, handle potential zeros
    lst_kelvin = convert_b10toKelvin(original_lst)
    lst_kelvin[original_lst == 0] = np.nan # Mask out sensor errors for plotting

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Verification for Scene: {date_str}", fontsize=16)

    im1 = axes[0].imshow(lst_kelvin, cmap='viridis')
    axes[0].set_title("Original LST Scene (Context)")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', pad=0.05, label="Temperature (K)")

    im2 = axes[1].imshow(final_mask, cmap='gray')
    axes[1].set_title("Final Processed Binary Mask")
    axes[1].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_filename = f"mask_verification_{date_str}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved verification plot to {output_filepath}")


# --- New Mask Tiling Function ---
def tile_and_save_masks(binary_mask, date_str, output_dir, tile_size=256, stride=128, min_mask_percent=10.0, max_mask_percent=80.0):
    """
    Tiles a binary mask and saves the tiles as PNG files if they meet coverage criteria.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    h, w = binary_mask.shape
    tile_count = 0
    
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            mask_tile = binary_mask[i:i+tile_size, j:j+tile_size]
            
            # --- Check the mask percentage of the INDIVIDUAL tile ---
            tile_mask_percentage = (np.sum(mask_tile) / mask_tile.size) * 100
            if not (min_mask_percent <= tile_mask_percentage <= max_mask_percent):
                continue  # Skip this tile if it's outside the desired range
            
            # Save the valid mask tile as a PNG file.
            img_tile = Image.fromarray((mask_tile * 255).astype(np.uint8))  # uint8 2-D array -> 'L' mode
            
            tile_filename = f"mask_{date_str}_tile_{i}_{j}.png"
            output_filepath = output_dir / tile_filename
            img_tile.save(output_filepath)
            tile_count += 1

    if tile_count > 0:
        print(f"Saved {tile_count} valid mask tiles for scene {date_str}.")


# --- Main Processing Logic ---
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate Training Masks")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Define output directory for the mask tiles
    landsat_path = Path(config['paths']['landsat_dir'])
    output_dir_masks = Path(config['paths']['masks_out'])
    output_dir_masks.mkdir(parents=True, exist_ok=True)
    params = config['parameters']

    b10_files = find_landsat_files(landsat_path, "T1_ST_B10.TIF")
    qa_files = {p.name.split('_')[3]: p for p in find_landsat_files(landsat_path, "T1_QA_PIXEL.TIF")}
    print(f"Found {len(b10_files)} ST_B10 files in {landsat_path}")

    # Process all scenes in the directory
    for b10_path in b10_files:
        date_str = b10_path.name.split('_')[3]
        if date_str not in qa_files: continue
        
        qa_path = qa_files[date_str]
        print(f"--- Processing scene for masks: {date_str} ---")
        
        try:
            ls_bounds, ls_crs, ls_trans, ls_h, ls_w = get_raster_bounds(b10_path)
            with rio.open(b10_path) as src: ls_data = src.read(1)
            with rio.open(qa_path) as src: qa_data = src.read(1)

            # --- Create the comprehensive binary mask ---
            masks = qa_masks(qa_data)
            qa_bad_pixels = masks['cloud'] | masks['shadow'] | masks['cirrus'] | masks['dilated'] | masks['snow']
            sensor_error_pixels = (ls_data == 0)
            
            binary_mask = (qa_bad_pixels | sensor_error_pixels).astype(np.uint8)
            
            # --- Apply the same geometric transformations as the main script ---
            # Set no-data boundary to NaN for cropping
            mask_with_nan = np.where(qa_data == 1, np.nan, binary_mask.astype(np.float32))

            valid_data_mask = (qa_data != 1)
            polygons = [shape(geom) for geom, val in shapes(valid_data_mask.astype(np.uint8), mask=valid_data_mask, transform=ls_trans) if val == 1]
            if not polygons: continue
            merged_poly = unary_union(polygons)
            angle = polygon_rotation_angle(merged_poly)

            # Use the 2D mask directly with the transformation functions
            unrotated_mask, _ = crop_and_unrotate(mask_with_nan, merged_poly, angle, ls_trans)
            if unrotated_mask.size == 0: continue
            
            trimmed_mask, _, _ = trim_nan_edges_limited(unrotated_mask, border=100)
            if trimmed_mask.size == 0: continue
            
            final_binary_mask = np.nan_to_num(trimmed_mask).astype(np.uint8)

            # --- TILE AND SAVE MASKS ---
            tile_and_save_masks(final_binary_mask, date_str, output_dir_masks,
                                tile_size=params['tile_size'], stride=params['stride'],
                                min_mask_percent=params['mask_min_percent'],
                                max_mask_percent=params['mask_max_percent'])
            
            gc.collect()

        except Exception as e:
            print(f"Failed to process {date_str}: {e}")

    print("All scenes processed for mask generation.")

