"""
Fills the real cloud / shadow gaps of Landsat LST scenes with a trained model
and writes georeferenced GeoTIFFs (original CRS / transform).

Usage (from the repository root):
    python src/inference.py --config configs/config.yaml
"""
import argparse
import yaml
import os
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.transform import rotate
from datetime import datetime
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
import gc
from scipy.ndimage import uniform_filter
import torch
import rasterio as rio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from itertools import islice


from models import build_generator  # src/models.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_landsat_files(folder, suffix):
    """Case-insensitive search (Landsat files end in .TIF; Linux globbing is case-sensitive)."""
    suffix = suffix.upper()
    return sorted(p for p in Path(folder).iterdir() if p.name.upper().endswith(suffix))


def plot_results(normalized_stack, final_mask, run_id, output_dir):
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
    fig.suptitle(f"Processed Scene: {run_id}", fontsize=16)

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
    plot_filename = f"plot_256_{run_id}.svg"
    output_filepath = output_dir / plot_filename
    #plt.savefig(output_filepath)
    plt.savefig(output_filepath.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_filepath}")

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
        print("--> Low NaN count detected. Plotting visible markers...")
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


# --- Data Normalization Functions ---

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

def normalize_slope(slope_array,max_slope):
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
    if array.ndim == 3:
        valid_2d = ~np.isnan(array).all(axis=0)
    else:
        valid_2d = ~np.isnan(array)

    if not np.any(valid_2d):
        return np.array([]), transform, None

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
        minr, minc, maxr, maxc = 0, 0, unrotated.shape[-2], unrotated.shape[-1]
        unrotated_cropped = unrotated

    # --- Remember the coordinates so we can reverse this later ---
    restore_info = {
        'bbox1': (min_row, max_row, min_col, max_col),
        'bbox2': (minr, maxr, minc, maxc),
        'unrotated_shape': unrotated.shape,
        'angle_deg': angle_deg
    }

    return unrotated_cropped, transform, restore_info

def reverse_crop_and_unrotate(inpainted_trimmed, top_idx, left_idx, restore_info, orig_h, orig_w):
    """Reverses the cropping and rotation to place the tile back in original geographic space."""
    min_row, max_row, min_col, max_col = restore_info['bbox1']
    minr, maxr, minc, maxc = restore_info['bbox2']
    angle_deg = restore_info['angle_deg']
    unrotated_shape = restore_info['unrotated_shape']

    # 1. Un-Trim
    H_trimmed, W_trimmed = inpainted_trimmed.shape
    H_uncropped2, W_uncropped2 = (maxr - minr), (maxc - minc)
    untrimmed = np.full((H_uncropped2, W_uncropped2), np.nan)
    untrimmed[top_idx:top_idx+H_trimmed, left_idx:left_idx+W_trimmed] = inpainted_trimmed

    # 2. Un-Crop Step 2
    unrotated_2d_shape = (unrotated_shape[1], unrotated_shape[2]) if len(unrotated_shape) == 3 else unrotated_shape
    unrotated_full = np.full(unrotated_2d_shape, np.nan)
    unrotated_full[minr:maxr, minc:maxc] = untrimmed

    # 3. Reverse the Rotation
    rotated_back = rotate(unrotated_full, angle=-(90-angle_deg), resize=False, order=0, mode='constant', cval=np.nan)

    # 4. Un-Crop Step 1 (Paste into full scene)
    full_scene = np.full((orig_h, orig_w), np.nan)
    patch_h, patch_w = rotated_back.shape
    full_scene[min_row:min_row+patch_h, min_col:min_col+patch_w] = rotated_back

    return full_scene

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

def plot_final_scene(original_lst, mask, inpainted_lst, scene_name, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Actual Inpainting: {scene_name}", fontsize=16)

    vmin, vmax = np.nanmin(inpainted_lst), np.nanmax(inpainted_lst)

    axes[0].imshow(original_lst, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original LST (with Clouds)")

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Actual Cloud Mask")

    axes[2].imshow(inpainted_lst, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2].set_title("Inpainted LST")

    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"result_{scene_name}.png", dpi=300)
    plt.close(fig)

def generate_tiles_adaptive(scene_h, scene_w, tile_size=1024, stride=512):
    """
    Generates all possible tile coordinates from the scene.
    """
    tiles = []

    # Calculate ranges for iteration
    r_coords = list(range(0, scene_h - tile_size + 1, stride))
    if (scene_h - tile_size) % stride != 0 and scene_h > tile_size:
        r_coords.append(scene_h - tile_size)
    elif scene_h <= tile_size:
        r_coords = [0]

    c_coords = list(range(0, scene_w - tile_size + 1, stride))
    if (scene_w - tile_size) % stride != 0 and scene_w > tile_size:
        c_coords.append(scene_w - tile_size)
    elif scene_w <= tile_size:
        c_coords = [0]

    # Ensure coordinates are unique
    r_coords = sorted(list(set(r_coords)))
    c_coords = sorted(list(set(c_coords)))

    for r in r_coords:
        for c in c_coords:
            tiles.append({
                'r_start': r,
                'c_start': c,
                'r_end': r + tile_size,
                'c_end': c + tile_size
            })
    return tiles

def get_gaussian_window(size, sigma_scale=0.25):
    """
    Creates a 2D Gaussian window to weigh center pixels higher than edges.
    sigma_scale: Controls width. 0.25 concentrates weight in the center, avoiding edge artifacts.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x_grid, y_grid = np.meshgrid(x, y)
    d = np.sqrt(x_grid**2 + y_grid**2)

    # Gaussian formula
    g = np.exp(-(d**2 / (2.0 * sigma_scale**2)))

    # Normalize to 0-1 range
    return g.astype(np.float32)

def inpaint_tile(model, device, data_tile_np, mask_tile_np):
    """
    Inpaints a single tile using our specific LaMaGenerator model.

    Args:
        model: Our loaded LaMaGenerator.
        device: Torch device.
        data_tile_np (np.ndarray): 20-channel normalized data tile [0, 1].
        mask_tile_np (np.ndarray): 1-channel mask tile (0=valid, 1=masked).

    Returns:
        np.ndarray: Inpainted 1-channel LST tile [0, 1].
    """
    # Combine data and mask to create the 21-channel model input
    model_input_np = np.vstack([data_tile_np, mask_tile_np])

    # Convert to tensor, add batch dim, send to device
    input_tensor = torch.from_numpy(model_input_np).unsqueeze(0).to(device)

    with torch.no_grad():
        # Run the model
        output_tensor = model(input_tensor)

    # Convert back to numpy, remove batch/channel dims
    output_lst_tile_np = output_tensor.squeeze().cpu().numpy()

    return output_lst_tile_np
def calculate_valid_percentage_from_mask(mask_tile):
    """
    Calculates the percentage of valid (non-masked) pixels in a mask tile.
    Assumes mask is 0=valid, 1=masked (hole).
    """
    if mask_tile.size == 0:
        return 0.0
    # 0 is valid, 1 is masked. We count the valid pixels.
    valid_pixels = np.sum(mask_tile == 0)
    return (valid_pixels / mask_tile.size) * 100


def adaptive_inpainting(model, device, full_data_stack_np, full_mask_np,
                        tile_size=256, min_valid_percentage=5, max_iterations=15):
    """
    Fixed Adaptive Inpainting with Batch Updating and Gaussian Blending.
    Removes grid artifacts by averaging overlapping tile predictions.
    """

    # --- Setup ---
    current_data_stack = full_data_stack_np.copy()
    current_mask = full_mask_np.copy()              # (1, H, W)
    current_mask_2d = current_mask.squeeze()        # (H, W) - 1=Hole, 0=Valid

    # We will update this specific channel (LST)
    current_lst = current_data_stack[0]

    C, H, W = current_data_stack.shape
    stride = tile_size // 2  # 50% overlap is required for blending

    # Pre-calculate the weight window
    gaussian_weights = get_gaussian_window(tile_size, sigma_scale=0.25)

    # Tracking arrays for visualization/debugging
    iteration_map = np.zeros_like(current_mask_2d, dtype=np.int16)
    iteration_map[current_mask_2d == 1] = -1
    tile_process_order = {}
    iteration_1_order_data = []

    print("\n--- Starting Gaussian Blended Inpainting ---")
    print(f"  - Scene: {H}x{W}, Tile: {tile_size}, Stride: {stride}")

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # --- BATCH BUFFERS (Crucial for Blending) ---
        # These accumulate predictions from ALL overlapping tiles before we update the main image
        accum_prediction = np.zeros((H, W), dtype=np.float32)
        accum_weight = np.zeros((H, W), dtype=np.float32)

        # 1. Generate Tiles
        all_tile_coords = generate_tiles_adaptive(H, W, tile_size, stride)

        # 2. Filter Tiles
        tiles_to_process = []
        for coords in all_tile_coords:
            r_start, r_end = coords['r_start'], coords['r_end']
            c_start, c_end = coords['c_start'], coords['c_end']

            mask_tile = current_mask_2d[r_start:r_end, c_start:c_end]

            # Check if this tile covers any remaining holes
            has_masked_pixels = np.any(mask_tile == 1)

            if has_masked_pixels:
                # Calculate valid percentage
                valid_perc = calculate_valid_percentage_from_mask(mask_tile)

                if valid_perc >= min_valid_percentage:
                    coords['valid_percentage'] = valid_perc
                    tiles_to_process.append(coords)

        # Sort tiles (Highest valid % first)
        tiles_to_process.sort(key=lambda x: x['valid_percentage'], reverse=True)

        if iteration == 1:
            iteration_1_order_data = tiles_to_process

        if not tiles_to_process:
            print("  - No suitable tiles found this iteration.")
            if np.sum(current_mask_2d == 1) > 0:
                print("  - Warning: Holes remain but no tiles met the valid % criteria.")
            break

        print(f"  - Processing {len(tiles_to_process)} tiles...")

        # 3. Process Tiles (Accumulation Phase)
        for tile_info in tiles_to_process:
            r_start, r_end = tile_info['r_start'], tile_info['r_end']
            c_start, c_end = tile_info['c_start'], tile_info['c_end']

            # Extract Data (using current global state)
            data_tile_np = current_data_stack[:, r_start:r_end, c_start:c_end]
            mask_tile_np = current_mask[:, r_start:r_end, c_start:c_end]

            # Predict
            inpainted_tile = inpaint_tile(model, device, data_tile_np, mask_tile_np)

            # --- BLENDING LOGIC ---
            # Identify which pixels in this tile are *currently* holes
            # We ONLY accumulate predictions for the holes.
            local_mask_2d = mask_tile_np.squeeze()
            active_holes = (local_mask_2d == 1)

            if np.any(active_holes):
                # Get the corresponding weights (handle edge cases where tile is smaller than window)
                h_t, w_t = active_holes.shape
                local_weights = gaussian_weights[:h_t, :w_t]

                # Accumulate weighted prediction
                accum_prediction[r_start:r_end, c_start:c_end][active_holes] += \
                    (inpainted_tile[active_holes] * local_weights[active_holes])

                # Accumulate weights
                accum_weight[r_start:r_end, c_start:c_end][active_holes] += \
                    local_weights[active_holes]

                # Track process order
                if (r_start, c_start) not in tile_process_order:
                    tile_process_order[(r_start, c_start)] = iteration

        # 4. Update Phase (Batch Update)
        # Find pixels that received at least one prediction
        updated_mask = (accum_weight > 0)

        if np.sum(updated_mask) == 0:
            print("  - No progress made in this iteration.")
            break

        # Calculate weighted average
        final_values = np.zeros_like(current_lst)
        # Avoid division by zero
        final_values[updated_mask] = accum_prediction[updated_mask] / accum_weight[updated_mask]

        # --- CRITICAL: Update the Global State HERE, not inside the loop ---

        # 1. Update LST values
        current_lst[updated_mask] = final_values[updated_mask]

        # 2. Update Mask (Mark as valid)
        current_mask_2d[updated_mask] = 0

        # 3. Update Visuals
        iteration_map[updated_mask] = iteration

        print(f"  - Updated {np.sum(updated_mask)} pixels.")

        # 5. Convergence Check
        remaining_holes = np.sum(current_mask_2d == 1)
        if remaining_holes == 0:
            print("🎉 All holes inpainted!")
            break

    return current_lst, tile_process_order, iteration_1_order_data, iteration_map

def plot_inpaint_category_map(iteration_map, scene_name, output_dir, max_iterations):
    """
    Generates a plot showing the final category of each pixel:
    - Original Data
    - Skipped (Never Inpainted)
    - Inpainted (Iteration 1, 2, 3...)
    """
    print(f"  - Visualizing final pixel category map for {scene_name}...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12 * (iteration_map.shape[0] / iteration_map.shape[1])))

    # --- Define Colors and Labels ---
    # We use 'turbo' to get distinct colors for iterations
    iter_cmap = plt.get_cmap('turbo', max_iterations)  # plt.cm.get_cmap was removed in Matplotlib 3.9

    # Base colors
    color_list = [
        '#FF0000',  # Red (for -1, Skipped)
        '#606060'   # Dark Gray (for 0, Original)
    ]
    label_list = [
        'Skipped (Hole)',
        'Original Data'
    ]
    # Boundaries for -1 and 0
    bounds = [-1.5, -0.5, 0.5]

    # Dynamically add colors/labels for iterations that actually happened
    max_iter_found = np.max(iteration_map)
    for i in range(1, max_iter_found + 1):
        color_list.append(iter_cmap((i - 1) / max(1, max_iterations - 1)))  # colour from 'turbo'
        label_list.append(f'Inpainted Iter {i}')
        bounds.append(f'{i}.5')

    bounds = [float(b) for b in bounds] # Convert boundaries to float

    # Create the discrete colormap and normalizer
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(bounds, cmap.N)
    # --- End Define Colors ---

    # Plot the map
    im = ax.imshow(iteration_map, cmap=cmap, norm=norm)

    # Add a colorbar
    # Calculate tick positions to be in the middle of each color
    tick_positions = [b + 0.5 for b in bounds[:-1]]
    cbar = fig.colorbar(im, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(label_list, fontsize=10)

    ax.set_title(f'Pixel Inpainting Category Map for {scene_name}', fontsize=16)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Save the figure
    plot_filename = f"plot_inpainting_CATEGORY_MAP_{scene_name}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"  - Saved category map plot to {output_filepath}")

def plot_inpainting_order(scene_h, scene_w, tile_process_order, tile_size, scene_name, output_dir, max_iterations):
    """
    Generates a plot visualizing the iteration number in which
    each tile was first processed.
    """
    print(f"  - Visualizing inpainting process order for {scene_name}...")

    if not tile_process_order:
        print("  - No process order data to plot. Skipping.")
        return

    # 1. Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10 * (scene_h / scene_w)))

    # 2. Draw the main scene boundary
    scene_boundary = plt.Rectangle((0, 0), scene_w, scene_h,
                                   facecolor='gray', alpha=0.1, edgecolor='black',
                                   linewidth=1, label=f'Scene Boundary ({scene_h}x{scene_w})')
    ax.add_patch(scene_boundary)

    # 3. Get a colormap, from 1 to max_iterations
    cmap = plt.get_cmap('turbo', max_iterations)  # plt.cm.get_cmap was removed in Matplotlib 3.9
    norm = plt.Normalize(vmin=1, vmax=max_iterations)

    # 4. Draw each tile, colored by its iteration number
    for (r_start, c_start), iteration in tile_process_order.items():

        color = cmap(norm(iteration))

        tile_rect = plt.Rectangle((c_start, r_start), tile_size, tile_size,
                                  facecolor=color,
                                  alpha=0.6,
                                  edgecolor='none') # No edges for a cleaner look
        ax.add_patch(tile_rect)

    # 5. Finalize plot
    ax.set_title(f'Inpainting Process Order for {scene_name}\n'
                 f'(Color = First Iteration Tile was Blended)',
                 fontsize=14)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')

    # Set limits and invert Y-axis
    ax.set_xlim(0, scene_w)
    ax.set_ylim(scene_h, 0)
    ax.set_aspect('equal')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=range(1, max_iterations + 1))
    cbar.set_label('Iteration Number')

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # 6. Save the figure
    plot_filename = f"plot_inpainting_ORDER_{scene_name}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"  - Saved inpainting order plot to {output_filepath}")


def load_model_for_testing(ckpt_path, input_channels, output_channels, device, config=None):
    """
    Loads a saved generator checkpoint (FFC-LaMa or Swin-UNet, see config['generator']) for inference.
    """
    print(f"Loading checkpoint from: {ckpt_path}")

    # 1. Load the checkpoint file
    #    map_location loads the model onto the correct device
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {ckpt_path}")
        return None
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None

    # 2. Initialize a new model instance
    #    It MUST have the same architecture as the saved one
    model = build_generator(config or {}, in_channels=input_channels, out_channels=output_channels).to(device)

    # 3. Extract the generator's weights from the checkpoint
    if 'generator_state_dict' not in checkpoint:
        print("Error: Checkpoint is missing 'generator_state_dict'.")
        print("It might be an invalid or incomplete checkpoint file.")
        return None

    generator_weights = checkpoint['generator_state_dict']

    # 4. Load the weights into the model
    try:
        model.load_state_dict(generator_weights)
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This often means the model architecture (e.g., input_channels) defined")
        print("in this script does not match the one in the checkpoint.")
        return None

    # 5. Set the model to evaluation mode
    #    This disables things like dropout or batchnorm updates
    model.eval()

    print("Model loaded successfully and set to evaluation mode.")
    return model


def plot_inpaint_progression(iteration_map, scene_name, output_dir):
    """
    Generates a 4-panel "storyboard" plot showing the mask (holes)
    shrinking at different key iterations.
    """
    print(f"  - Visualizing inpainting progression storyboard for {scene_name}...")

    # Create the 2x2 plot
    fig, axes = plt.subplots(2, 2, figsize=(18, 18 * (iteration_map.shape[0] / iteration_map.shape[1])))
    fig.suptitle(f'Inpainting Progression for {scene_name}\n(White = Holes/Mask)', fontsize=20)

    # --- Panel 1: Original Mask ---
    # Show all pixels that were not original data (i.e., all holes)
    original_mask = (iteration_map != 0)
    axes[0, 0].imshow(original_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('1. Original Mask (All Holes)', fontsize=14)

    # --- Panel 2: After Iteration 1 ---
    # Show pixels that were skipped (-1) OR required more than 1 iteration (> 1)
    after_iter_1_mask = (iteration_map == -1) | (iteration_map > 1)
    axes[0, 1].imshow(after_iter_1_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('2. Holes Remaining After Iteration 1', fontsize=14)

    # --- Panel 3: After Iteration 5 (a middle step) ---
    # Show pixels that were skipped (-1) OR required more than 5 iterations (> 5)
    after_iter_5_mask = (iteration_map == -1) | (iteration_map > 5)
    axes[1, 0].imshow(after_iter_5_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('3. Holes Remaining After Iteration 5', fontsize=14)

    # --- Panel 4: Final Skipped Pixels ---
    # Show only the pixels that were skipped and *never* inpainted
    final_skipped_mask = (iteration_map == -1)
    axes[1, 1].imshow(final_skipped_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('4. Final Skipped Pixels (Never Inpainted)', fontsize=14)

    # Clean up axes
    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plot_filename = f"plot_inpainting_PROGRESSION_{scene_name}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"  - Saved progression storyboard plot to {output_filepath}")



# --- Main Processing Logic ---
if __name__ == '__main__':
    print(f"Loading LaMa model on {DEVICE}...")
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Run LaMa LST Inpainting Inference")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()

    # 2. Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    CHECKPOINT_PATH = Path(config['checkpoint_path'])
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {CHECKPOINT_PATH}")

    landsat_path = Path(config['landsat_dir'])
    LULC_Path = Path(config['lulc_dir'])
    DEM_Path = Path(config['dem_path'])
    SLOPE_Path = Path(config['slope_path'])
    ASPECT_Path = Path(config['aspect_path'])
    NDVI_Path = Path(config['ndvi_dir'])
    HILLSHADE_Path = Path(config['hillshade_dir'])
    SAR_Path = Path(config['sar_dir'])

    # Notice we use different output keys for the production inference script
    output_dir_plots = Path(config.get('inference_output_plots', 'outputs/actual_inpainted_scenes'))
    output_dir_tiles = Path(config.get('inference_output_tiles', 'outputs/actual_inpainted_scenes_tiles'))
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    output_dir_tiles.mkdir(parents=True, exist_ok=True)

    # 4. Dynamically calculate input channels based on the config
    ACTIVE_INDICES = config.get('active_data_indices', list(range(20)))
    assert ACTIVE_INDICES[0] == 0, "active_data_indices must start with 0 (LST channel)"
    active_count = len(ACTIVE_INDICES)
    INPUT_CHANNELS = active_count + 1
    OUTPUT_CHANNELS = 1

    model = load_model_for_testing(CHECKPOINT_PATH, INPUT_CHANNELS, OUTPUT_CHANNELS, DEVICE, config)
    if model is None:
        raise SystemExit(f"Could not load a model from {CHECKPOINT_PATH}")


    print("Scanning NDVI directory and all subdirectories...")
    ndvi_files = list(NDVI_Path.rglob('*_NDVI.tif'))
    ndvi_by_date = {datetime.strptime(fp.name.split('_')[3][:8], '%Y%m%d'): fp for fp in ndvi_files}
    available_ndvi_dates = list(ndvi_by_date.keys())
    print(f"Found {len(available_ndvi_dates)} NDVI files across all folders.")


    print("Scanning LULC directory ...")
    lulc_tiffs = [str(LULC_Path / i) for i in os.listdir(LULC_Path) if i.endswith('.tif')]
    lulc_by_year = {}
    for fp in lulc_tiffs:
        try:
            year = Path(fp).name.split('_')[-1].split('-')[0][:4]
            if year.isdigit(): lulc_by_year.setdefault(year, []).append(fp)
        except IndexError: continue
    available_lulc_years = list(lulc_by_year.keys())

    print("Scanning Hillshade directory ...")
    hillshade_files = {tuple(p.name.split('_')[0:4]): p for p in HILLSHADE_Path.glob('*_hillshade.tif')}

    print("Scanning SAR directory ...")
    sar_files = {p.name.split('_')[5][:-4]: p for p in SAR_Path.glob('*.tif')}


    LULC_CLASSES = [1, 2, 4, 5, 7, 8, 9, 10, 11]

    print("Loading DEM ...")
    with rio.open(DEM_Path) as dem_src:
        dem_profile = dem_src.profile
        dem_elevation = dem_src.read(1).astype(np.float32)
    print("Loading Slope ...")
    with rio.open(SLOPE_Path) as slope_src:
        dem_slope = slope_src.read(1).astype(np.float32)

    print("Loading Aspect ...")
    with rio.open(ASPECT_Path) as aspect_src:
        dem_aspect = aspect_src.read(1).astype(np.float32)
    print("DEM, slope, and aspect loaded successfully.")

    print("Scanning LST directory ...")



    b10_files = find_landsat_files(landsat_path, "T1_ST_B10.TIF")
    qa_files = {p.name.split('_')[3]: p for p in find_landsat_files(landsat_path, "T1_QA_PIXEL.TIF")}

    print(f"Found {len(b10_files)} scenes to process.")



    for b10_path in b10_files:
        date_str = b10_path.name.split('_')[3]
        scene_id = b10_path.name.split('_T1_ST_B10')[0]
        run_id = f"{date_str}"
        print(f"\nProcessing Scene: {scene_id}")

        if date_str not in qa_files:
            print(f"  - Missing QA file for {date_str}, skipping.")
            continue

        qa_path = qa_files[date_str]

        # 1. Load Base Landsat & Mask
        ls_bounds, ls_crs, ls_trans, ls_h, ls_w = get_raster_bounds(b10_path)
        with rio.open(b10_path) as src: ls_data = src.read(1).astype(np.float32)
        lst_kelvin_original = convert_b10toKelvin(ls_data)

        with rio.open(qa_path) as src: qa_data = src.read(1)
        masks = qa_masks(qa_data)

        # Create the ACTUAL mask (No artificial insertion)
        qa_bad_pixels = masks['cloud'] | masks['shadow'] | masks['cirrus'] | masks['dilated'] | masks['snow']
        sensor_error_pixels = (ls_data == 0)
        binary_mask_real = (qa_bad_pixels | sensor_error_pixels).astype(np.uint8)

        # Check if the scene is completely cloudy or completely clear
        missing_perc = (np.sum(binary_mask_real) / binary_mask_real.size) * 100
        print(f"  - Actual Cloud Cover: {missing_perc:.2f}%")

        if missing_perc == 0:
            print("  - Scene is perfectly clear. Skipping inpainting.")
            continue
        if missing_perc > 95:
            print("  - Scene is too cloudy to reconstruct accurately. Skipping.")
            continue

        # --- Match Hillshade ---
        # 1. Extract key identifiers from the thermal filename
        filename_parts = b10_path.name.split('_')
        path_row = filename_parts[2]       # e.g., '193026'
        acquisition_date_str = filename_parts[3] # e.g., '20250531'
        acquisition_date = datetime.strptime(date_str, '%Y%m%d')

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



        ls_bounds, ls_crs, ls_trans, ls_h, ls_w = get_raster_bounds(b10_path)

        # Load LST data to be inpainted
        with rio.open(b10_path) as src: ls_data = src.read(1).astype(np.float32)
        lst_kelvin_original = convert_b10toKelvin(ls_data)


        #  Load *original* QA mask
        with rio.open(qa_files[date_str]) as src: qa_data = src.read(1)

        masks = qa_masks(qa_data)
        qa_bad_pixels = masks['cloud'] | masks['shadow'] | masks['cirrus'] | masks['dilated'] | masks['snow']
        sensor_error_pixels = (ls_data == 0)
        binary_mask=(qa_bad_pixels | sensor_error_pixels).astype(np.uint8)

        doy_channel = np.full((ls_h, ls_w), acquisition_date.timetuple().tm_yday, dtype=np.uint16)



        ground_truth = convert_b10toKelvin(ls_data)
        ground_truth[binary_mask==1]=0

        stack_ground_truth=np.stack([ground_truth,
                        binary_mask])


        # Load hillsahade data
        with rio.open(hillshade_path) as src:  hillshade_data = src.read(1); hillshade_profile = src.profile

        print("Reprojecting hillshade data to match Landsat grid...")
        hillshade_reproj= np.zeros((ls_h, ls_w), dtype=np.float32)

        reproject(
            source=hillshade_data,
            destination=hillshade_reproj,
            src_transform=hillshade_profile['transform'],
            src_crs=hillshade_profile['crs'],
            dst_transform=ls_trans,
            dst_crs=ls_crs,
            resampling=Resampling.nearest) # Use nearest for circular data




        # --- Load SAR or create zero array ---
        sar_path = sar_files.get(date_str)
        if sar_path:
            print(f"  - Found matching SAR file: {sar_path.name}")
            with rio.open(sar_path) as src:
                sar_data_vv = src.read(1).astype(np.float32)
                sar_data_vh = src.read(2).astype(np.float32)
                sar_profile = src.profile
        else:
            print(" - No SAR file found for this date. Filling with zeros.")
            #sar_data_vv = np.zeros((ls_h, ls_w), dtype=np.float32)
            sar_data_vv = np.full((ls_h, ls_w), -25.0, dtype=np.float32)
            #sar_data_vh = np.zeros((ls_h, ls_w), dtype=np.float32)
            sar_data_vh = np.full((ls_h, ls_w), -30.0, dtype=np.float32)
            # For zero-filled data, we can't get a profile, but reprojection won't happen anyway
            sar_profile = None


        print("Reprojecting SAR Data data to match Landsat grid...")

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




        # Find nearest available LULC, Load LULC of the nearest year, and Reproject
        lulc_year = find_nearest_year(str(acquisition_date.year), available_lulc_years)
        if not lulc_year: continue

        print(f"Loading and Reprojecting LULC data for year {lulc_year}...")

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

        # Find nearest NDVI, Load nearest NDVI and Reproject
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

        lst_kelvin_run = lst_kelvin_original.copy()
        lst_kelvin_run[binary_mask == 1] = 0


        stack = np.stack([
            lst_kelvin_run,
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
        stack_ground_truth[:, qa_data == 1] = np.nan

        valid_data_mask = (qa_data != 1)
        polygons = [shape(geom) for geom, val in shapes(valid_data_mask.astype(np.uint8), mask=valid_data_mask, transform=ls_trans) if val == 1]
        if not polygons: continue
        merged_poly = unary_union(polygons)
        angle = polygon_rotation_angle(merged_poly)

        # unrotated_stack, _ = crop_and_unrotate(stack, merged_poly, angle, ls_trans)
        # unrotated_stack_groud_truth, _ = crop_and_unrotate(stack_ground_truth, merged_poly, angle, ls_trans)

        unrotated_stack, _, restore_info = crop_and_unrotate(stack, merged_poly, angle, ls_trans)
        unrotated_stack_groud_truth, _, _ = crop_and_unrotate(stack_ground_truth, merged_poly, angle, ls_trans)

        if unrotated_stack.size == 0: continue

        ## commenteed to get rid of the error appearing afte adapting the adaptive inpainting function
        # trimmed_stack, _, _ = trim_nan_edges_limited(unrotated_stack, border=100)
        # trimmed_stack_gt, _, _ = trim_nan_edges_limited(unrotated_stack_groud_truth, border=100)
        # if trimmed_stack.size == 0: continue

        trimmed_stack_all, top_idx, left_idx = trim_nan_edges_limited(unrotated_stack, border=100)
        if trimmed_stack_all.size == 0: continue

        H_trimmed, W_trimmed = trimmed_stack_all.shape[1], trimmed_stack_all.shape[2]

        # 2. APPLY THE EXACT SAME SLICE TO THE GROUND TRUTH STACK
        trimmed_stack_gt = unrotated_stack_groud_truth[
            :,
            top_idx:top_idx + H_trimmed,
            left_idx:left_idx + W_trimmed
        ]

        # 3. Separate the data and mask from the now-unified stacks
        unnormalized_data = trimmed_stack_all[:-1]
        final_binary_mask = np.nan_to_num(trimmed_stack_all[-1]).astype(np.uint8)


        # Extract the Ground Truth LST and mask it based on the UNIFIED trimmed stack.
        gt = trimmed_stack_gt[0].copy()
        # The mask here is the original scene mask (channel 1 of the GT stack)
        gt[trimmed_stack_gt[1] == 1] = np.nan

        gt_profile = {
           'driver': 'GTiff',
           'height': gt.shape[0], # CORRECT: use .shape[0] for height
           'width': gt.shape[1],  # CORRECT: use .shape[1] for width
           'count': 1,
           'dtype': gt.dtype,
           'compress': 'lzw',
           # NO CRS or TRANSFORM - Georeferencing is lost
           'nodata': np.nan
        }

        # --- Save the 'ground_truth' array for validation (as .npy) ---
        gt_save_path = output_dir_tiles / f"{run_id}_ground_truth.tif"
        with rio.open(gt_save_path, 'w', **gt_profile) as dst:
            dst.write(gt.astype(np.float32), 1)

        # --- MODIFIED: Normalize NDVI data ---
        norm_cfg = config['normalization']

        normalized_lst = normalize_lst(unnormalized_data[0], norm_cfg['lst']['min'], norm_cfg['lst']['max'])
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
            print("sar data were interpolated")
        else:
            normalized_sar_vv = normalize_sar(unnormalized_data[8], norm_cfg['sar_vv']['min'], norm_cfg['sar_vv']['max'])
            normalized_sar_vh = normalize_sar(unnormalized_data[9], norm_cfg['sar_vh']['min'], norm_cfg['sar_vh']['max'])

        if nan_percentage_sar_vv > 10:
            print("SAR data have large gaps (>10% NaN)")
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

        plot_results(normalized_stack, final_binary_mask, run_id, output_dir_plots)


        # Feed the model only the channels it was trained with (config: active_data_indices).
        # LST must stay at index 0 because adaptive_inpainting() updates channel 0.
        model_stack = normalized_stack[ACTIVE_INDICES]
        if model:
            # Pass the selected data channels and the 1-channel (squeezed) mask
            current_max_iterations=20
            inpainted_lst_normalized, process_order_data, iter_1_order, final_iteration_map = adaptive_inpainting(
                model=model,
                device=DEVICE,
                full_data_stack_np=model_stack,  # (len(active_data_indices), H, W)
                full_mask_np=final_binary_mask[np.newaxis, ...], # (1, H, W)
                tile_size=256,
                min_valid_percentage=5,
                max_iterations=current_max_iterations
            )


            if final_iteration_map is not None:
                plot_inpaint_category_map(
                    final_iteration_map,
                    run_id,
                    output_dir_plots,
                    current_max_iterations
                )

                plot_inpaint_progression(
                    final_iteration_map,
                    run_id,
                    output_dir_plots
                )



            # --- Save .npy files (rotated/trimmed) ---
            if inpainted_lst_normalized is not None:
                # --- Un-normalize the final LST ---
                min_k = config['normalization']['lst']['min']
                max_k = config['normalization']['lst']['max']
                inpainted_lst_kelvin = (inpainted_lst_normalized * (max_k - min_k)) + min_k


               # 2. Reverse the spatial transformations!
                print("  - Reversing rotation and placing back into original scene coordinates...")
                full_scene_inpainted = reverse_crop_and_unrotate(
                    inpainted_lst_kelvin, top_idx, left_idx, restore_info, ls_h, ls_w
                )

                # (removed an unused normalize_lst(x) call without min/max that raised a TypeError)

                # 3. Merge intelligently!
                final_full_product = lst_kelvin_original.copy()

                # Pre-wipe all original clouds and sensor errors to NaN
                final_full_product[binary_mask == 1] = np.nan

                # 4. Paste LaMa's predictions back into the valid holes!
                valid_prediction_mask = (binary_mask == 1) & ~np.isnan(full_scene_inpainted)
                final_full_product[valid_prediction_mask] = full_scene_inpainted[valid_prediction_mask]

                # ==========================================
                # --- YOUR RULE: KEEP THE EXCLUDED ROWS OUT ---
                # Since full_scene_inpainted is empty (NaN) in the 100-pixel margin we trimmed,
                # this forces the final image to permanently drop that margin too.
                final_full_product[np.isnan(full_scene_inpainted)] = np.nan
                # ==========================================

                # 5. Guarantee the massive Landsat exterior collar is completely wiped
                final_full_product[qa_data == 1] = np.nan


                # 4. Generate the full-scene side-by-side plot
                print("  - Generating final side-by-side full scene plot...")
                plot_final_scene(
                    original_lst=lst_kelvin_original,
                    mask=binary_mask_real,
                    inpainted_lst=final_full_product,
                    scene_name=run_id,
                    out_dir=output_dir_plots
                )

                # 5. Save with TRUE Georeferencing
                final_output_profile = {
                    'driver': 'GTiff',
                    'height': ls_h,
                    'width': ls_w,
                    'count': 1,
                    'dtype': final_full_product.dtype,
                    'crs': ls_crs,
                    'transform': ls_trans,
                    'compress': 'lzw',
                    'nodata': np.nan
                }

                tiff_save_path = output_dir_tiles / f"{run_id}_inpainted_final_256_v2.tif"
                print(f"  - Saving geographically accurate GeoTIFF to: {tiff_save_path.name}")
                with rio.open(tiff_save_path, 'w', **final_output_profile) as dst:
                    dst.write(final_full_product.astype(np.float32), 1)


        gc.collect()

    print("\n--- All real scenes processed. ---")