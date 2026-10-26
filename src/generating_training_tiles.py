"""
Builds the multi-channel training tiles (GeoTIFF, float32) from Landsat LST and
the auxiliary layers (LULC, DOY, DEM/slope/aspect, NDVI, hillshade, Sentinel-1).

Channel layout of every tile (0-based):
    0      normalised LST
    1-9    one-hot LULC (Esri classes 1, 2, 4, 5, 7, 8, 9, 10, 11)
    10-11  DOY sin / cos
    12     normalised elevation
    13     normalised slope
    14-15  aspect sin / cos
    16     normalised NDVI
    17     normalised hillshade
    18-19  normalised Sentinel-1 VV / VH
    20     binary mask placeholder (all zeros for clear-sky training tiles)

Usage (from the repository root):
    python src/generating_training_tiles.py --config configs/data_config.yaml
"""
import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import yaml
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
from skimage.transform import rotate


def find_landsat_files(folder, suffix):
    """Case-insensitive search (Landsat files end in .TIF; Linux globbing is case-sensitive)."""
    suffix = suffix.upper()
    return sorted(p for p in Path(folder).iterdir() if p.name.upper().endswith(suffix))
# --- Data Normalization Functions ---


def fix_raster_file(data):
    """
    Opens a raster, interpolates NaN values, and saves the cleaned
    file to the specified output directory.
    """

    nan_mask = np.isnan(data)
    nan_count_before = np.sum(nan_mask)

    # --- EFFICIENT INTERPOLATION METHOD ---
    data_for_sum = np.where(nan_mask, 0, data)
    valid_pixel_mask = (~nan_mask).astype(float)

    sum_of_neighbors = uniform_filter(data_for_sum, size=3)
    count_of_neighbors = uniform_filter(valid_pixel_mask, size=3)
    
    count_of_neighbors = np.where(count_of_neighbors == 0, 1, count_of_neighbors)

    mean_of_neighbors = sum_of_neighbors / count_of_neighbors

    cleaned_data = data.copy()
    cleaned_data[nan_mask] = mean_of_neighbors[nan_mask]
    # --- END OF EFFICIENT METHOD ---

    final_nan_count = np.isnan(cleaned_data).sum()
    if final_nan_count > 0:
        np.nan_to_num(cleaned_data, copy=False, nan=0.0)

    print(f"  - Fixed! NaN count reduced from {nan_count_before} to {np.isnan(cleaned_data).sum()}.")
    
    return cleaned_data


def nan_counter(data):
    nan_count = np.isnan(data).sum()
    total_pixels = data.size
    nan_percentage = (nan_count / total_pixels) * 100 if total_pixels > 0 else 0
    
    return nan_percentage 
    

def visualize_nan_values(data, band_num=1):
    """
    Opens a raster file, plots it with NaN values highlighted,
    and adds visible markers if the NaN count is low.
    """
    data = data

    # --- Count NaN values ---
    nan_count = np.isnan(data).sum()
    total_pixels = data.size
    nan_percentage = (nan_count / total_pixels) * 100 if total_pixels > 0 else 0
    print(f"--> Found {nan_count} NaN pixels ({nan_percentage:.2f}%) in this band.")

    # This function creates a special array that "hides" NaN values.
    masked_data = np.ma.masked_invalid(data)

    # --- SETUP THE PLOT AND COLORMAP ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    cmap = plt.cm.viridis 
    # Use red for the underlying NaN pixels
    cmap.set_bad(color='red', alpha=1)

    im = ax.imshow(masked_data, cmap=cmap)

    # --- MODIFIED: Add NaN count to title ---
    title_text = (f'Raster Data with NaN values shown in RED\n'
                  f'NaN Count: {nan_count} ({nan_percentage:.2f}%)')
    ax.set_title(title_text)
    
    # --- Add highly visible markers for a small number of NaNs ---
    # If the count is low, plot large circles on each NaN location.
    if 0 < nan_count < 1000: # Set a reasonable limit to avoid clutter
        print(f"--> Low NaN count detected. Plotting visible markers...")
        # Find the (row, col) coordinates of all NaN values
        rows, cols = np.where(np.isnan(data))
        
        # Plot a scatter point at each NaN location
        ax.scatter(cols, rows, s=80, c='magenta', edgecolor='black', 
                   marker='o', label=f'NaN Locations ({nan_count})')
        ax.legend() # Display the legend for the markers
    # -----------------------------------------------------------------
    
    fig.colorbar(im, ax=ax, label='Pixel Value')
    
    print("Displaying plot. Close the plot window to end the script.")
    plt.show()


        
def normalize_lst(lst_array, min_kelvin, max_kelvin):
    """Normalizes LST data from Kelvin to a [0, 1] range for UHI studies."""
    lst_clipped = np.clip(lst_array, min_kelvin, max_kelvin)
    normalized_lst = (lst_clipped - min_kelvin) / (max_kelvin - min_kelvin)
    return normalized_lst

def one_hot_encode_lulc(lulc_array, class_values):
    """Converts a single-channel LULC array into a multi-channel one-hot encoded array."""
    one_hot_channels = []
    for class_val in class_values:
        channel = (lulc_array == class_val).astype(np.uint8)
        one_hot_channels.append(channel)
    return np.stack(one_hot_channels, axis=0)

def normalize_doy(doy_array):
    """Normalizes Day of Year (DOY) using sine and cosine transformations."""
    doy_sin = np.sin(2 * np.pi * doy_array / 366.0).astype(np.float32)
    doy_cos = np.cos(2 * np.pi * doy_array / 366.0).astype(np.float32)
    return doy_sin, doy_cos


def normalize_elevation(elev_array, min_elev, max_elev):
    """Normalizes elevation data to a [0, 1] range using typical global values."""
    elev_clipped = np.clip(elev_array, min_elev, max_elev)
    return (elev_clipped - min_elev) / (max_elev - min_elev)

def normalize_slope(slope_array, max_slope):
    """Normalizes slope (in degrees) to a [0, 1] range."""
    # Slope from richdem is 0-90 degrees
    return slope_array / max_slope

def normalize_aspect(aspect_array):
    """Normalizes aspect (in degrees) using sine and cosine transformations."""
    # Aspect is circular (0-360 degrees), so sin/cos is best
    aspect_rad = np.deg2rad(aspect_array)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    return aspect_sin, aspect_cos

def normalize_ndvi(raw_ndvi_array, scale_factor, offset):
    """
    Applies the correct scale and offset to Copernicus GLS NDVI data,
    then normalizes it to a [0, 1] range for the model.
    """
    # These are the standard values for this Copernicus product

    # First, convert the raw integer data to the scientific NDVI value (-1 to 1)
    scientific_ndvi = raw_ndvi_array * scale_factor + offset
    
    # Clip to the theoretical range to handle any minor outliers
    scientific_ndvi_clipped = np.clip(scientific_ndvi, -1.0, 1.0)
    
    return (scientific_ndvi_clipped + 1.0) / 2.0

def normalize_hillshade(hillshade_array, max_val=255.0):
    """Normalizes hillshade data from [0, 255] to [0, 1]."""
    return hillshade_array / max_val

def normalize_sar(sar_array, min_db, max_db):
    sar_clipped = np.clip(sar_array, min_db, max_db)
    return (sar_clipped - min_db) / (max_db - min_db)


# --- Helper Functions ---

def convert_b10toKelvin(array):
    """Converts scaled ST_B10 pixel values to Kelvin."""
    return array * 0.00341802 + 149.0

def get_raster_bounds(raster_path):
    """Extracts the geographical bounds, CRS, transform, height, and width."""
    with rio.open(raster_path) as src:
        return src.bounds, src.crs, src.transform, src.height, src.width

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

def find_nearest_year(target_year, available_years):
    """Finds the nearest year in a list of available years."""
    available_years.sort()
    if not available_years: return None
    return min(available_years, key=lambda x: abs(int(x) - int(target_year)))

# --- User's Transformation Functions ---

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


def plot_results(normalized_stack, final_mask, date_str, output_dir):
    """Plots the final normalized data channels and the binary mask."""
    LULC_CLASSES_PLOT = [1, 2, 4, 5, 7, 8, 9, 10, 11]
    colors = ["#1A5BAB", "#358221", "#87D19E", "#FFDB5C", "#ED022A", "#EDE9E4", "#F2FAFF", "#C8C8C8", "#C6AD8D"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(LULC_CLASSES_PLOT + [LULC_CLASSES_PLOT[-1] + 1], cmap.N)
    
    lulc_plot = np.zeros(normalized_stack.shape[1:], dtype=np.uint8)
    for i, class_val in enumerate(LULC_CLASSES_PLOT):
        lulc_plot[normalized_stack[i+1] == 1] = class_val

      # --- Create a 8-panel figure ---
    fig, axes = plt.subplots(1, 9, figsize=(45, 5))
    fig.suptitle(f"Processed Scene: {date_str}", fontsize=16)

    axes[0].imshow(normalized_stack[0], cmap="viridis"); axes[0].set_title("Norm. LST")
    axes[1].imshow(lulc_plot, cmap=cmap, norm=norm); axes[1].set_title("LULC")
    axes[2].imshow(normalized_stack[12], cmap="terrain"); axes[2].set_title("Norm. Elevation")
    axes[3].imshow(normalized_stack[13], cmap="plasma"); axes[3].set_title("Norm. Slope")
    axes[4].imshow(normalized_stack[14], cmap="coolwarm"); axes[4].set_title("Aspect (Sine)")
    axes[5].imshow(normalized_stack[16], cmap="RdYlGn"); axes[5].set_title("Norm. NDVI")
    axes[6].imshow(normalized_stack[17], cmap="gray"); axes[6].set_title("Hillshade")
    axes[7].imshow(normalized_stack[18], cmap="bone"); axes[7].set_title("SAR VV")
    axes[8].imshow(final_mask, cmap="gray"); axes[8].set_title("Cloud Mask")
    
    for ax in axes: ax.axis("off")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure and close it
    output_filepath = output_dir / f"plot_{date_str}.png"
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)  # avoid accumulating open figures over many scenes
    print(f"Saved plot to {output_filepath}")



def tile_and_save_training_data(normalized_stack, binary_mask, date_str, output_dir, tile_size=256, stride=128,
                                min_kelvin=260.0, max_kelvin=342.0):
    """
    Tiles the normalized data and saves the clear-sky tiles for training as multi-channel TIFFs.
    The last channel of the TIFF will be the binary mask.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    _, h, w = normalized_stack.shape
    valid_tile_count = 0
    generated_coords = set()
    MIN_KELVIN, MAX_KELVIN = min_kelvin, max_kelvin
    
    for i in range(0, h - tile_size + 1, stride):
        for j in range(0, w - tile_size + 1, stride):
            data_tile = normalized_stack[:, i:i+tile_size, j:j+tile_size]
            mask_tile = binary_mask[i:i+tile_size, j:j+tile_size]
            
            # Condition 1: Check for any bad data in the mask tile.
            if np.any(mask_tile == 1):
                continue
            
            hillshade_channel = data_tile[-3]
            sar_vv_channel = data_tile[-2]
            sar_vh_channel = data_tile[-1]
            
            if np.isnan(hillshade_channel).any():
                print("Discovered NaN in hillshade, skipping tile")
                continue
            
            if np.isnan(sar_vv_channel).any():
                print("Discovered NaN in SAR VV, skipping tile")
                continue
            
            if np.isnan(sar_vh_channel).any():
                print("Discovered NaN in SAR VH, skipping tile")
                continue

            
            # Condition 2: Check LST range. Un-normalize to check Kelvin range.
            lst_tile_norm = data_tile[0]
            lst_tile_kelvin = lst_tile_norm * (MAX_KELVIN - MIN_KELVIN) + MIN_KELVIN
            
            if (np.max(lst_tile_kelvin) - np.min(lst_tile_kelvin)) <= 40:
                valid_tile_count += 1
                
                # --- Save the valid tile as a TIFF ---
                # Stack the mask as the last channel
                tile_to_save = np.vstack([data_tile, mask_tile[np.newaxis, ...].astype(data_tile.dtype)])
                
                tile_filename = f"scene_{date_str}_tile_{i}_{j}.tif"
                output_filepath = output_dir / tile_filename
                
                # Create metadata for the TIFF file
                profile = {
                    'driver': 'GTiff',
                    'height': tile_size,
                    'width': tile_size,
                    'count': tile_to_save.shape[0],
                    'dtype': tile_to_save.dtype,
                    'compress': 'lzw'
                }
                
                
                
                with rio.open(output_filepath, 'w', **profile) as dst:
                    dst.write(tile_to_save)
                
                generated_coords.add((i, j)) # Add the coordinate of this valid tile to our set

    print(f"Saved {valid_tile_count} valid grid-based training tiles for scene {date_str}.")
    return generated_coords

def generate_supplemental_random_tiles(normalized_stack, binary_mask, date_str, output_dir, existing_coords,
                                      num_new_tiles=300, tile_size=256, min_kelvin=260.0, max_kelvin=342.0, rng=None):
    """
    Generates new random tiles that are unique from the grid tiles AND from each other.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = rng if rng is not None else np.random.default_rng()

    _, h, w = normalized_stack.shape
    new_tile_count = 0
    attempts = 0
    max_attempts = num_new_tiles * 10
    MIN_KELVIN, MAX_KELVIN = min_kelvin, max_kelvin

    while new_tile_count < num_new_tiles and attempts < max_attempts:
        attempts += 1
        
        rand_i = int(rng.integers(0, h - tile_size + 1))
        rand_j = int(rng.integers(0, w - tile_size + 1))

        if (rand_i, rand_j) in existing_coords:
            continue
        
        # ... (rest of the validation logic is the same) ...
        data_tile = normalized_stack[:, rand_i:rand_i+tile_size, rand_j:rand_j+tile_size]
        mask_tile = binary_mask[rand_i:rand_i+tile_size, rand_j:rand_j+tile_size]
        if np.any(mask_tile == 1):
            continue
        
        hillshade_channel = data_tile[-3]
        sar_vv_channel = data_tile[-2]
        sar_vh_channel = data_tile[-1]
        
        if np.isnan(hillshade_channel).any():
            print("Discovered NaN in hillshade, skipping tile")
            continue
        
        if np.isnan(sar_vv_channel).any():
            print("Discovered NaN in SAR VV, skipping tile")
            continue
        
        if np.isnan(sar_vh_channel).any():
            print("Discovered NaN in SAR VH, skipping tile")
            continue
        
        lst_tile_norm = data_tile[0]
        lst_tile_kelvin = lst_tile_norm * (MAX_KELVIN - MIN_KELVIN) + MIN_KELVIN
        if (np.max(lst_tile_kelvin) - np.min(lst_tile_kelvin)) > 40:
            continue
            
        # --- Save the tile ---
        tile_to_save = np.vstack([data_tile, mask_tile[np.newaxis, ...].astype(data_tile.dtype)])
        tile_filename = f"scene_{date_str}_rand_tile_{rand_i}_{rand_j}.tif"
        profile = {'driver': 'GTiff', 'height': tile_size, 'width': tile_size, 'count': tile_to_save.shape[0], 'dtype': tile_to_save.dtype, 'compress': 'lzw'}
        
        with rio.open(output_dir / tile_filename, 'w', **profile) as dst:
            dst.write(tile_to_save)
        
        new_tile_count += 1
        
        # --- THE FIX --- ✨
        # Add the new coordinate to the set to avoid repeating it in subsequent random draws.
        existing_coords.add((rand_i, rand_j))

    print(f"Saved {new_tile_count} new unique supplemental random tiles for scene {date_str}.")

# --- Main Processing Logic ---
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Generate multi-channel training tiles")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
        
    landsat_path =  Path(config['paths']['landsat_dir'])
    LULC_Path =  Path(config['paths']['lulc_dir'])
    DEM_Path = Path(config['paths']['dem_path']) 
    SLOPE_Path=Path(config['paths']['slope_path'])
    ASPECT_Path=Path(config['paths']['aspect_path'])
    NDVI_Path = Path(config['paths']['ndvi_dir'])
    HILLSHADE_Path = Path(config['paths']['hillshade_dir'])
    SAR_Path = Path(config['paths']['sar_dir'])
    
    # Define output directories
    output_dir_tiles = Path(config['paths']['tiles_out'])
    output_dir_plots = Path(config['paths']['plots_out'])
    output_dir_tiles.mkdir(parents=True, exist_ok=True)
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    params = config['parameters']
    tile_size = params['tile_size']
    stride = params['stride']
    n_random = params['supplemental_random_tiles']
    rng = np.random.default_rng(params.get('random_seed', 42))  # reproducible random tiles
    
    
    print("Scanning NDVI directory and all subdirectories...")
    ndvi_files = list(NDVI_Path.rglob('*_NDVI.tif'))
    ndvi_by_date = {datetime.strptime(fp.name.split('_')[3][:8], '%Y%m%d'): fp for fp in ndvi_files}
    available_ndvi_dates = list(ndvi_by_date.keys())
    print(f"Found {len(available_ndvi_dates)} NDVI files across all folders.")
    

    lulc_tiffs = [str(LULC_Path / i) for i in os.listdir(LULC_Path) if i.endswith('.tif')]
    lulc_by_year = {}
    for fp in lulc_tiffs:
        try:
            year = Path(fp).name.split('_')[-1].split('-')[0][:4]
            if year.isdigit(): lulc_by_year.setdefault(year, []).append(fp)
        except IndexError: continue
    available_lulc_years = list(lulc_by_year.keys())
    
    hillshade_files = {tuple(p.name.split('_')[0:4]): p for p in HILLSHADE_Path.glob('*_hillshade.tif')}
    #sar_files = {p.name.split('_')[6]: p for p in SAR_Path.glob('landsat_*_S1_stitch_to_footprint.tif')}
    sar_files = {p.name.split('_')[5][:-4]: p for p in SAR_Path.glob('*.tif')}
    LULC_CLASSES = [1, 2, 4, 5, 7, 8, 9, 10, 11]    
    
    print("Loading DEM data from files...")
    with rio.open(DEM_Path) as dem_src:
        dem_profile = dem_src.profile
        dem_elevation = dem_src.read(1).astype(np.float32)
        
    with rio.open(SLOPE_Path) as slope_src:
        dem_slope = slope_src.read(1).astype(np.float32)

    with rio.open(ASPECT_Path) as aspect_src:
        dem_aspect = aspect_src.read(1).astype(np.float32)
    print("DEM, slope, and aspect loaded successfully.")
    

    b10_files = find_landsat_files(landsat_path, "T1_ST_B10.TIF")
    qa_files = {p.name.split('_')[3]: p for p in find_landsat_files(landsat_path, "T1_QA_PIXEL.TIF")}
    print(f"Found {len(b10_files)} ST_B10 files in {landsat_path}")


    for b10_path in b10_files: 
        date_str = b10_path.name.split('_')[3]
        if date_str not in qa_files: continue
        
        qa_path = qa_files[date_str]
        print(f"--- Processing scene: {date_str} ---")
        
        
        # --- Match Hillshade ---
        # 1. Extract key identifiers from the thermal filename
        filename_parts = b10_path.name.split('_')
        path_row = filename_parts[2]       # e.g., '193026'
        acquisition_date_str = filename_parts[3] # e.g., '20250531'

        # 2. Create a flexible search pattern using wildcards (*)
        search_pattern = f"*{path_row}_{acquisition_date_str}*_hillshade.tif"
        
        # 3. Search for the matching file in the hillshade directory
        matching_files = list(HILLSHADE_Path.glob(search_pattern))

        # 4. Check the result
        if matching_files:
            hillshade_path = matching_files[0] # Take the first match
            print(f"  - Found matching hillshade: {hillshade_path.name}")
        else:
            # If no match, print the warning and skip to the next scene
            print(f"  - WARNING: No matching hillshade found for {b10_path.name} using pattern '{search_pattern}'. Skipping scene.")
            continue

        # --- Match SAR (with zero-filling for missing dates) ---
        sar_path = sar_files.get(date_str)
        
        ls_bounds, ls_crs, ls_trans, ls_h, ls_w = get_raster_bounds(b10_path)
        with rio.open(b10_path) as src: ls_data = src.read(1).astype(np.float32)
        with rio.open(qa_files[date_str]) as src: qa_data = src.read(1)

        acquisition_date = datetime.strptime(date_str, '%Y%m%d')
        
        #--- Load data and profiles for reprojection ---
        print("  - Loading source data for reprojection...")
        # --- REVISED TWO-STEP DATE MATCHING LOGIC ---
        # Step 1: Find the closest year available in the NDVI dataset.
        landsat_year_str = str(acquisition_date.year)
        available_ndvi_years = sorted(list(set([str(d.year) for d in available_ndvi_dates])))
        
        # Use the helper function to find the best matching year
        best_year_str = find_nearest_year(landsat_year_str, available_ndvi_years)
        best_year = int(best_year_str)
        print(f"Landsat year is {landsat_year_str}. Closest available NDVI year is {best_year}.")

        # Step 2: From the dates within that best year, find the one closest to the Landsat day-of-year.
        dates_in_best_year = [d for d in available_ndvi_dates if d.year == best_year]
        
        if not dates_in_best_year:
            print(f"Error: No NDVI dates found for the determined best year '{best_year}'. Skipping scene.")
            continue

        landsat_doy = acquisition_date.timetuple().tm_yday
        
        # This function calculates the shortest difference in days, handling the year-end wrap-around
        def seasonal_difference(d):
            ndvi_doy = d.timetuple().tm_yday
            diff = abs(landsat_doy - ndvi_doy)
            return min(diff, 366 - diff) # 366 is safe for leap years
        
        # Find the date within the best_year list that has the smallest seasonal difference
        nearest_ndvi_date = min(dates_in_best_year, key=seasonal_difference)
        # --------------------------------------------------
        
        ndvi_filepath = ndvi_by_date[nearest_ndvi_date]
        print(f"Landsat date: {acquisition_date.date()}, Matched NDVI date: {nearest_ndvi_date.date()}")
        
    
        # ndvi_filepath = ndvi_by_date[nearest_ndvi_date]

        with rio.open(ndvi_filepath) as ndvi_src:
            ndvi_data = ndvi_src.read(1).astype(np.float32)
            ndvi_profile = ndvi_src.profile
        # -------------------------------------------------------------------------
        
        with rio.open(hillshade_path) as src:
            hillshade_data = src.read(1)
            hillshade_profile = src.profile
        
        # --- Load SAR or create zero array ---
        if sar_path:
            print(f"  - Found matching SAR file: {sar_path.name}")
            with rio.open(sar_path) as src:
                sar_data_vv = src.read(1).astype(np.float32)
                sar_data_vh = src.read(2).astype(np.float32)
                sar_profile = src.profile
        else:
            # No Sentinel-1 stitch for this date: fill with the lower normalisation bound (-> 0 after normalisation)
            print(f"  - No SAR file found for this date. Filling with the minimum dB value (normalised 0).")
            #sar_data_vv = np.zeros((ls_h, ls_w), dtype=np.float32)
            sar_data_vv = np.full((ls_h, ls_w), -25.0, dtype=np.float32)
            #sar_data_vh = np.zeros((ls_h, ls_w), dtype=np.float32)
            sar_data_vh = np.full((ls_h, ls_w), -30.0, dtype=np.float32)
            # For zero-filled data, we can't get a profile, but reprojection won't happen anyway
            sar_profile = None 

        lulc_year = find_nearest_year(str(acquisition_date.year), available_lulc_years)
        if not lulc_year: continue
        
        print(f"Reprojecting LULC data for year {lulc_year}...")
        
        lulc_reprojected = np.zeros((ls_h, ls_w), dtype=np.uint8)
        lulc_sources_paths = lulc_by_year[lulc_year]

        for lulc_path in lulc_sources_paths:
            with rio.open(lulc_path) as src_lulc:
                temp_tile_reprojected = np.zeros_like(lulc_reprojected)
                reproject(
                    source=rio.band(src_lulc, 1),
                    destination=temp_tile_reprojected,
                    src_transform=src_lulc.transform,
                    src_crs=src_lulc.crs,
                    dst_transform=ls_trans,
                    dst_crs=ls_crs,
                    resampling=Resampling.nearest
                )
                lulc_reprojected = np.where(temp_tile_reprojected > 0, temp_tile_reprojected, lulc_reprojected)
        
        
        
        print("Reprojecting NDVI data to match Landsat grid...")
        ndvi_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)
        reproject(
            source=ndvi_data,
            destination=ndvi_reproj,
            src_transform=ndvi_profile['transform'],
            src_crs=ndvi_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.bilinear
        )
        
        
        print("Reprojecting DEM data to match Landsat grid...")
        dem_elevation_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)
        dem_slope_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)
        dem_aspect_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)

        # Reproject Elevation
        reproject(
            source=dem_elevation,
            destination=dem_elevation_reproj,
            src_transform=dem_profile['transform'],
            src_crs=dem_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.bilinear)
        
        # Reproject Slope
        reproject(
            source=dem_slope,
            destination=dem_slope_reproj,
            src_transform=dem_profile['transform'],
            src_crs=dem_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.bilinear)

        # Reproject Aspect
        reproject(
            source=dem_aspect,
            destination=dem_aspect_reproj,
            src_transform=dem_profile['transform'],
            src_crs=dem_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.nearest) # Use nearest for circular data
        
        
        print("Reprojecting hillshade data to match Landsat grid...")
        hillshade_reproj= np.zeros((ls_h, ls_w), dtype=np.float32)
       
        reproject(
            source=hillshade_data,
            destination=hillshade_reproj,
            src_transform=hillshade_profile['transform'],
            src_crs=hillshade_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.nearest)

        print("Reprojecting SAR data to match Landsat grid...")
        
        if sar_path and sar_profile:
            sar_vv_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)
            reproject(
                source=sar_data_vv,
                destination=sar_vv_reproj,
                src_transform=sar_profile['transform'],
                src_crs=sar_profile['crs'],
                dst_transform=ls_trans,
                dst_crs=ls_crs,
                resampling=Resampling.nearest) # Use nearest for circular data
            
            sar_vh_reproj = np.zeros((ls_h, ls_w), dtype=np.float32)
            reproject(
                source=sar_data_vh,
                destination=sar_vh_reproj,
                src_transform=sar_profile['transform'],
                src_crs=sar_profile['crs'],
                dst_transform=ls_trans,
                dst_crs=ls_crs,
                resampling=Resampling.nearest) # Use nearest for circular data
        else:
            # If SAR is zero-filled, it's already in the target grid, so we can just copy it.
            sar_vv_reproj = sar_data_vv
            sar_vh_reproj = sar_data_vh
        
        
        lst_kelvin = convert_b10toKelvin(ls_data)
        doy_channel = np.full((ls_h, ls_w), acquisition_date.timetuple().tm_yday, dtype=np.uint16)
        
        masks = qa_masks(qa_data)
        qa_bad_pixels = masks['cloud'] | masks['shadow'] | masks['cirrus'] | masks['dilated'] | masks['snow']
        sensor_error_pixels = (ls_data == 0)
        
        binary_mask = (qa_bad_pixels | sensor_error_pixels).astype(np.uint8)
        lst_kelvin[binary_mask == 1] = 0
        

        stack = np.stack([
            lst_kelvin, 
            lulc_reprojected, 
            doy_channel, 
            dem_elevation_reproj, 
            dem_slope_reproj, 
            dem_aspect_reproj, 
            ndvi_reproj,
            hillshade_reproj,
            sar_vv_reproj,
            sar_vh_reproj,
            binary_mask
        ])
        

        
        #stack = np.stack([lst_kelvin, lulc_reprojected, doy_channel, binary_mask])
        stack[:, qa_data == 1] = np.nan

        valid_data_mask = (qa_data != 1)
        polygons = [shape(geom) for geom, val in shapes(valid_data_mask.astype(np.uint8), mask=valid_data_mask, transform=ls_trans) if val == 1]
        if not polygons: continue
        merged_poly = unary_union(polygons)
        angle = polygon_rotation_angle(merged_poly)

        unrotated_stack, _ = crop_and_unrotate(stack, merged_poly, angle, ls_trans)
        if unrotated_stack.size == 0: continue
        
        trimmed_stack, _, _ = trim_nan_edges_limited(unrotated_stack, border=100)
        if trimmed_stack.size == 0: continue
        
        unnormalized_data = trimmed_stack[:-1]
        final_binary_mask = np.nan_to_num(trimmed_stack[-1]).astype(np.uint8)
        
        
        
        # --- MODIFIED: Normalize NDVI data ---
        norm_cfg = config['normalization']
        normalized_lst = normalize_lst(unnormalized_data[0],norm_cfg['lst']['min'], norm_cfg['lst']['max'])
        one_hot_lulc = one_hot_encode_lulc(unnormalized_data[1], LULC_CLASSES)
        doy_sin, doy_cos = normalize_doy(unnormalized_data[2])
        normalized_elev = normalize_elevation(unnormalized_data[3], norm_cfg['elevation']['min'], norm_cfg['elevation']['max'])
        normalized_slope = normalize_slope(unnormalized_data[4], norm_cfg['slope']['max'])
        aspect_sin, aspect_cos = normalize_aspect(unnormalized_data[5])
        normalized_ndvi = normalize_ndvi(unnormalized_data[6], norm_cfg['ndvi']['scale_factor'], norm_cfg['ndvi']['offset'])
        normalized_hillshade=normalize_hillshade(unnormalized_data[7])
        
        
        nan_percentage_sar_vv=nan_counter(unnormalized_data[8])
         
        if nan_percentage_sar_vv > 0 and nan_percentage_sar_vv <=10:
            #visualize_nan_values(unnormalized_data[9])
            cleaned_vv_data=fix_raster_file(unnormalized_data[8])
            cleaned_vh_data=fix_raster_file(unnormalized_data[9])
            #visualize_nan_values(cleaned_vh_data)
            normalized_sar_vv = normalize_sar(cleaned_vv_data, norm_cfg['sar_vv']['min'], norm_cfg['sar_vv']['max'])
            normalized_sar_vh = normalize_sar(cleaned_vh_data, norm_cfg['sar_vh']['min'], norm_cfg['sar_vh']['max'])
            print("SAR data were interpolated")
        else:
            normalized_sar_vv = normalize_sar(unnormalized_data[8], norm_cfg['sar_vv']['min'], norm_cfg['sar_vv']['max'])
            normalized_sar_vh = normalize_sar(unnormalized_data[9], norm_cfg['sar_vh']['min'], norm_cfg['sar_vh']['max'])

        if nan_percentage_sar_vv > 10:
            print("SAR data have large gaps; tiles containing NaN will be skipped")
        elif nan_percentage_sar_vv == 0:
            print("SAR data are NaN-free")

        # --- Add normalized DEM channels to the final stack ---
        normalized_stack = np.vstack([
            normalized_lst[np.newaxis, ...],
            one_hot_lulc,
            doy_sin[np.newaxis, ...],
            doy_cos[np.newaxis, ...],
            normalized_elev[np.newaxis, ...],
            normalized_slope[np.newaxis, ...],
            aspect_sin[np.newaxis, ...],
            aspect_cos[np.newaxis, ...],
            normalized_ndvi[np.newaxis, ...],
            normalized_hillshade[np.newaxis, ...],
            normalized_sar_vv[np.newaxis, ...],
            normalized_sar_vh[np.newaxis, ...],
        ]).astype(np.float32)
        

        print(f"Final normalized stack shape: {normalized_stack.shape}, Mask shape: {final_binary_mask.shape}")
        
        
        # np.savez_compressed(output_filepath, data=normalized_stack, mask=final_binary_mask)
        # print(f"Saved processed scene to: {output_filepath}\n")
        
        plot_results(normalized_stack, final_binary_mask, date_str, output_dir_plots)
        print("Plot saved")
        # --- TILE AND SAVE TRAINING DATA ---
        lst_min, lst_max = norm_cfg['lst']['min'], norm_cfg['lst']['max']
        grid_tile_coords = tile_and_save_training_data(
            normalized_stack, final_binary_mask, date_str, output_dir_tiles,
            tile_size=tile_size, stride=stride, min_kelvin=lst_min, max_kelvin=lst_max)

        # Supplemental random tiles (unique w.r.t. the grid tiles and each other)
        generate_supplemental_random_tiles(
            normalized_stack, final_binary_mask, date_str, output_dir_tiles,
            existing_coords=grid_tile_coords,
            num_new_tiles=n_random, tile_size=tile_size,
            min_kelvin=lst_min, max_kelvin=lst_max, rng=rng)
    
        
    print("All scenes processed.")

