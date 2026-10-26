"""
Quantitative evaluation on held-out Landsat scenes: real QA masks from other
dates (binned by cloud cover) are superimposed on test scenes, the gaps are
filled with the adaptive, Gaussian-blended tiling strategy, and MAE / RMSE / R
(overall and per LST quartile) are computed on the artificially hidden pixels.

Usage (from the repository root):
    python src/evaluate.py --config configs/config.yaml
"""
import os
import argparse
import yaml
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.transform import rotate
from datetime import datetime
from itertools import islice
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label, regionprops
import gc
from scipy.ndimage import uniform_filter
import re
import random
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import seaborn as sns
import csv
import pandas as pd
import rasterio as rio
from rasterio.features import shapes
from rasterio.warp import reproject, Resampling
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Import the model definition from your other file
# This assumes 'lama_model.py' is in the same directory
from models import build_generator  # src/models.py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_landsat_files(folder, suffix):
    """Case-insensitive search (Landsat files end in .TIF; Linux globbing is case-sensitive)."""
    suffix = suffix.upper()
    return sorted(p for p in Path(folder).iterdir() if p.name.upper().endswith(suffix))



def calibrate_dynamic_lulc_factors(lama_clear_lst, original_lst, cloud_mask, shadow_mask, lulc_2d):
    """
    Empirically derives the Apparent Thermal Inertia (ATI) factor for each LULC class
    by calculating the exact temperature deficit inside local clear-sky shadows.
    """
    print("  - Dynamically calibrating LULC thermal inertia from local shadows...")

    valid_shadow = shadow_mask & (original_lst > 0)

    # Calculate the exact temperature drop caused by the shadow
    delta_t_shadow = lama_clear_lst[valid_shadow] - original_lst[valid_shadow]

    unique_classes = np.unique(lulc_2d[valid_shadow])
    class_delta_t = {}

    for lulc_class in unique_classes:
        # Extract the temperature drops for this specific material
        class_pixels = delta_t_shadow[lulc_2d[valid_shadow] == lulc_class]

        # We need a statistically significant sample size to trust the physics
        if len(class_pixels) > 100:
            mean_drop = np.nanmean(class_pixels)
            if mean_drop > 0: # Physics check: shadows must cool the surface
                class_delta_t[lulc_class] = mean_drop

    # Normalize the factors (0.0 to 1.0) based on the material that lost the most heat
    if class_delta_t:
        max_drop = max(class_delta_t.values())
        calibrated_factors = {k: max(0.3, v / max_drop) for k, v in class_delta_t.items()}
        print(f"    -> Dynamically learned LULC factors: { {k: round(v, 2) for k, v in calibrated_factors.items()} }")
        return calibrated_factors, max_drop
    else:
        print("    -> WARNING: Insufficient shadows. Falling back to theoretical defaults.")
        # Default literature fallback if scene is 100% cloudy
        fallback = {1: 0.45, 2: 0.55, 4: 0.75, 5: 0.75, 7: 0.55, 8: 1.0, 9: 0.45, 10: 0.75, 11: 0.45}
        return fallback, 6.0 # Assume 6.0K max drop

def calculate_solar_insolation_multiplier(latitude_deg, doy, hour_local):
    """Calculates cos(Solar Zenith Angle) safely for arrays."""
    lat_rad = np.radians(latitude_deg)
    declination_deg = 23.45 * np.sin(np.radians((360.0 / 365.0) * (doy - 81.0)))
    dec_rad = np.radians(declination_deg)
    hour_angle_deg = 15.0 * (hour_local - 12.0)
    hour_rad = np.radians(hour_angle_deg)

    cos_sza = (np.sin(lat_rad) * np.sin(dec_rad)) + \
              (np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad))
    return np.maximum(0.0, cos_sza)

def generate_rse_all_weather_lst(
    lama_clear_lst, original_lst, cloud_mask, shadow_mask,
    sar_vv, hillshade, dem_elevation, raw_lulc_2d,
    latitude, doy, hour
):
    # ==========================================================
    # 1. DYNAMIC APPARENT THERMAL INERTIA (ATI)
    # ==========================================================
    # A. Get the dynamically calibrated LULC factors and the max observed penalty
    dynamic_lulc_factors, max_base_penalty = calibrate_dynamic_lulc_factors(
        lama_clear_lst, original_lst, cloud_mask, shadow_mask, raw_lulc_2d
    )

    # Map factors to the 2D grid
    base_ati = np.ones_like(raw_lulc_2d, dtype=np.float32)
    for lulc_class, factor in dynamic_lulc_factors.items():
        base_ati[raw_lulc_2d == lulc_class] = factor

    # B. Dynamic SAR Moisture Adjustment (Reduces penalty if soil is wet)
    moisture_modifier = 1.0 - (0.3 * np.clip(sar_vv, 0, 1))

    # Final localized thermal resistance
    final_ati_factor = base_ati * moisture_modifier

    # ==========================================================
    # 2. SHORTWAVE DEFICIT (Orbital & Topographic)
    # ==========================================================
    cos_sza = calculate_solar_insolation_multiplier(latitude, doy, hour)

    # How much solar radiation was lost? (Max * Orbit * Hillshade)
    shortwave_loss = max_base_penalty * cos_sza * np.clip(hillshade, 0, 1)

    # Scale the loss by the physical resistance of the surface
    net_shortwave_penalty = shortwave_loss * final_ati_factor

    # ==========================================================
    # 3. LONGWAVE SURPLUS (The Cloud Blanket Effect)
    # ==========================================================
    # Clouds emit heat back down.
    # Base warming of ~2.5K, decaying at higher altitudes (-6.5K per 1000m)
    LAPSE_RATE = 0.0065
    baseline_elev = np.nanmin(dem_elevation)

    longwave_warming = 2.5 - ((dem_elevation - baseline_elev) * LAPSE_RATE)
    longwave_warming = np.clip(longwave_warming, 0, None)

    # ==========================================================
    # 4. FINAL COMPOSITE
    # ==========================================================
    # Final Cooling = (Shortwave Lost) - (Longwave Gained)
    final_cooling_penalty = np.clip(net_shortwave_penalty - longwave_warming, 0, None)

    all_weather_lst = np.copy(original_lst)
    mask_bool = cloud_mask.astype(bool)

    # Subtract the physics-derived penalty from the LaMa clear-sky baseline
    all_weather_lst[mask_bool] = lama_clear_lst[mask_bool] - final_cooling_penalty[mask_bool]

    return all_weather_lst




# --- Validation Metrics Function ---
def calculate_and_print_metrics(y_true_with_nans, y_pred_full,final_binary_mask,scene_name,output_dir):
    """
    Calculates and prints validation metrics (MAE, RMSE, R) for the
    inpainted pixels, including a quantile analysis.
    """
    print(f"\n  --- Validation Metrics for {scene_name} ---")


    plot_validation_results(y_true_with_nans, y_pred_full, final_binary_mask, scene_name, output_dir)
    print(len(y_pred_full.flatten()))
    print(len(y_true_with_nans.flatten()))

    mask_bool=final_binary_mask.astype(bool)
    inpainted_pixels = y_pred_full[mask_bool]
    original_pixels = y_true_with_nans[mask_bool]
    # Flatten arrays for easier indexing
    y_true_flat = original_pixels.flatten()
    y_pred_flat = inpainted_pixels.flatten()

    print(len(y_true_flat))
    print(len(y_pred_flat))

    # Find the pixels that are valid in the ground truth
    # (i.e., the ones we artificially masked)
    validation_mask = ~np.isnan(y_true_flat)

    true_vals = y_true_flat[validation_mask]
    pred_vals = y_pred_flat[validation_mask]


    print(len(true_vals))
    print(len(pred_vals))

    if true_vals.size < 2: # Need at least 2 points for correlation
        print("  - Not enough validation pixels to calculate metrics.")
        return

    # --- 1. Overall Metrics ---
    mae = mean_absolute_error(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    corr, p_value = pearsonr(true_vals, pred_vals)

    print(f"  Overall (on {true_vals.size} pixels):")
    print(f"    - MAE:   {mae:.4f} (Kelvin)")
    print(f"    - RMSE:  {rmse:.4f} (Kelvin)")
    print(f"    - R:     {corr:.4f} (p-value: {p_value:.3g})")

    # --- 2. Quantile Analysis (based on ground truth LST) ---
    print("\n  Quantile Analysis (Error by LST):")

    metrics_to_return = {
        "validation_pixels": true_vals.size,
        "overall_mae": mae,
        "overall_rmse": rmse,
        "overall_r": corr
    }


    try:
        quantiles = np.percentile(true_vals, [25, 50, 75])
        q1_mask = true_vals < quantiles[0]
        q2_mask = (true_vals >= quantiles[0]) & (true_vals < quantiles[1])
        q3_mask = (true_vals >= quantiles[1]) & (true_vals < quantiles[2])
        q4_mask = true_vals >= quantiles[2]

        quantile_masks = {
            f"Q1 (Low LST, < {quantiles[0]:.2f} K)": q1_mask,
            f"Q2 ({quantiles[0]:.2f} - {quantiles[1]:.2f} K)": q2_mask,
            f"Q3 ({quantiles[1]:.2f} - {quantiles[2]:.2f} K)": q3_mask,
            f"Q4 (High LST, > {quantiles[2]:.2f} K)": q4_mask
        }

        for i, (q_name, q_mask) in enumerate(quantile_masks.items()):
            q_true = true_vals[q_mask]
            q_pred = pred_vals[q_mask]
            q_key = f"q{i+1}" # 'q1', 'q2', etc.
            if q_true.size > 0:
                q_mae = mean_absolute_error(q_true, q_pred)
                q_rmse = np.sqrt(mean_squared_error(q_true, q_pred))
                print(f"    - {q_name} ({q_true.size} pixels):")
                print(f"        - MAE: {q_mae:.4f}, RMSE: {q_rmse:.4f}")

                metrics_to_return[f"{q_key}_mae"] = q_mae
                metrics_to_return[f"{q_key}_rmse"] = q_rmse

            else:
                print(f"    - {q_name}: No pixels in this quantile.")
                metrics_to_return[f"{q_key}_mae"] = np.nan
                metrics_to_return[f"{q_key}_rmse"] = np.nan
    except Exception as e:
        print(f"    - Could not perform quantile analysis: {e}")
    print("  --- End of Metrics ---")

    return metrics_to_return


def plot_validation_results(y_true_with_nans, y_pred_full, final_binary_mask, scene_name, output_dir):
    """
    Generates a 4-panel comparison plot (Corrected for axis sharing):
    1. Masked Input Image (y_true masked by final_binary_mask)
    2. Inpainted Result (y_pred_full)
    3. Absolute Error Heatmap (on validation pixels)
    4. Histogram of Absolute Errors
    """
    print(f"  - Generating corrected 4-panel validation plot for {scene_name}...")

    # 1. --- Prepare data for all 4 panels ---

    # Panel 1: Masked Input Image
    masked_input = y_true_with_nans.copy()
    masked_input[final_binary_mask.astype(bool)] = np.nan

    # Panel 2: Inpainted Result
    inpainted_result = y_pred_full

    # Panel 3: Absolute Error Map
    abs_error_map = np.abs(inpainted_result - y_true_with_nans)

    # Panel 4: Histogram of Absolute Errors
    validation_pixel_mask_2d = ~np.isnan(y_true_with_nans) & (final_binary_mask == 1)

    if not np.any(validation_pixel_mask_2d):
        print("  - No validation pixels found (no artificial mask?). Skipping comparison plot.")
        return

    valid_true_vals = y_true_with_nans[validation_pixel_mask_2d]
    valid_pred_vals = inpainted_result[validation_pixel_mask_2d]
    abs_error_values = np.abs(valid_pred_vals - valid_true_vals)
    print("valid_pred_vals")
    print(len(valid_pred_vals))
    # 2. --- Determine common color ranges ---
    try:
        vmin = np.nanmin([np.nanmin(y_true_with_nans), np.nanmin(inpainted_result)])
        vmax = np.nanmax([np.nanmax(y_true_with_nans), np.nanmax(inpainted_result)])
    except ValueError:
        vmin, vmax = 270, 330

    # 3. --- Create the 2x2 plot ---

    # --- THIS IS THE FIRST FIX ---
    # Set sharex/sharey to False by default
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=False, sharey=False)
    fig.suptitle(f'Validation Comparison for {scene_name}', fontsize=20)

    # --- THIS IS THE SECOND FIX ---
    # Manually link the axes for the three image plots
    axes[0,1].sharex(axes[0,0])
    axes[0,1].sharey(axes[0,0])
    axes[1,0].sharex(axes[0,0])
    axes[1,0].sharey(axes[0,0])
    # Now axes[1,1] (the histogram) is independent

    # Panel 1: Masked Input
    im1 = axes[0,0].imshow(masked_input, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0,0].set_title('Masked Input (for Inpainting)', fontsize=14)
    fig.colorbar(im1, ax=axes[0,0], orientation='vertical', fraction=0.046, pad=0.04, label='LST (K)')
    axes[0,0].axis('off')

    # Panel 2: Inpainted Result
    im2 = axes[0,1].imshow(inpainted_result, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0,1].set_title('Inpainted Result (Full)', fontsize=14)
    fig.colorbar(im2, ax=axes[0,1], orientation='vertical', fraction=0.046, pad=0.04, label='LST (K)')
    axes[0,1].axis('off')

    # Panel 3: Absolute Error Map
    im3 = axes[1,0].imshow(abs_error_map, cmap='hot', vmin=0, vmax=10)
    axes[1,0].set_title('Absolute Error (on Validation Pixels)', fontsize=14)
    fig.colorbar(im3, ax=axes[1,0], orientation='vertical', fraction=0.046, pad=0.04, label='Absolute Error (K)')
    axes[1,0].axis('off')

    # Panel 4: Histogram of Absolute Errors
    ax_hist = axes[1,1]
    err_mean = np.nanmean(abs_error_values)
    err_std = np.nanstd(abs_error_values)

    sns.histplot(abs_error_values, kde=True, ax=ax_hist, bins=50, stat="density")
    ax_hist.axvline(err_mean, color='black', linestyle=':', linewidth=2, label=f"Mean = {err_mean:.3f}")
    ax_hist.set_title('Histogram of Absolute Errors', fontsize=14)
    ax_hist.set_xlabel('Absolute Error (K)')
    ax_hist.set_ylabel('Density')

    stats_text = f"Mean = {err_mean:.3f} K\nStd = {err_std:.3f} K\nN = {len(abs_error_values)}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax_hist.text(0.97, 0.97, stats_text, transform=ax_hist.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    ax_hist.legend(loc='upper right')

    # --- THIS IS THE THIRD FIX ---
    # The problematic lines are now removed.

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plot_filename = f"plot_validation_corrected_{scene_name}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"  - Saved corrected validation plot to {output_filepath}")

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



def visualize_tiling_strategy(scene_h, scene_w, tile_size, stride, scene_name, output_dir):
    """
    Generates a plot showing the adaptive tiling grid and overlaps,
    ZOOMED IN to a 3x3 grid of tiles in the CENTER of the scene.
    """
    print(f"  - Visualizing tiling strategy for {scene_name} (Center Zoom)...")

    # 1. Generate all tile coordinates
    all_tile_coords = generate_tiles_adaptive(scene_h, scene_w, tile_size, stride)

    if not all_tile_coords:
        print("  - No tiles were generated. Skipping plot.")
        return

    # 2. Find the geometric center of the scene
    scene_center_h = scene_h / 2
    scene_center_w = scene_w / 2

    # 3. Find the tile closest to the geometric center
    center_tile = None
    min_dist = float('inf')

    for coords in all_tile_coords:
        tile_center_w = coords['c_start'] + (coords['c_end'] - coords['c_start']) / 2
        tile_center_h = coords['r_start'] + (coords['r_end'] - coords['r_start']) / 2

        dist = np.sqrt((scene_center_w - tile_center_w)**2 + (scene_center_h - tile_center_h)**2)

        if dist < min_dist:
            min_dist = dist
            center_tile = coords

    if center_tile is None:
        print("  - Could not find center tile. Skipping plot.")
        return

    # 4. Find the 3x3 grid of tiles around the center tile
    center_tile_w = center_tile['c_start'] + (center_tile['c_end'] - center_tile['c_start']) / 2
    center_tile_h = center_tile['r_start'] + (center_tile['r_end'] - center_tile['r_start']) / 2

    # Select tiles whose centers are within ~1.5 tile sizes of the center tile
    plot_radius = tile_size * 1.5

    tiles_to_plot = []
    for coords in all_tile_coords:
        tile_center_w = coords['c_start'] + (coords['c_end'] - coords['c_start']) / 2
        tile_center_h = coords['r_start'] + (coords['r_end'] - coords['r_start']) / 2

        dist = np.sqrt((center_tile_w - tile_center_w)**2 + (center_tile_h - tile_center_h)**2)

        if dist < plot_radius:
            tiles_to_plot.append(coords)

    # 5. Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot scene boundary (faintly, for context)
    scene_boundary = plt.Rectangle((0, 0), scene_w, scene_h,
                                   facecolor='none', edgecolor='black', linewidth=1,
                                   linestyle='--', label=f'Scene Boundary ({scene_h}x{scene_w})')
    ax.add_patch(scene_boundary)

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # 6. Draw only the selected ~9 tiles
    for i, coords in enumerate(tiles_to_plot):
        r_start, r_end = coords['r_start'], coords['r_end']
        c_start, c_end = coords['c_start'], coords['c_end']
        tile_w, tile_h = c_end - c_start, r_end - r_start

        tile_rect = plt.Rectangle((c_start, r_start), tile_w, tile_h,
                                  facecolor='blue', alpha=0.1,
                                  edgecolor='blue', linewidth=1.5)
        ax.add_patch(tile_rect)

        ax.text(c_start + tile_w/2, r_start + tile_h/2, f'Tile\n({r_start}, {c_start})',
                 ha='center', va='center', color='red', fontsize=8, weight='bold')

        # Track limits for zooming
        if c_start < min_x: min_x = c_start
        if c_end > max_x: max_x = c_end
        if r_start < min_y: min_y = r_start
        if r_end > max_y: max_y = r_end

    # 7. Finalize plot
    ax.set_title(f'Tiling Strategy (Zoomed to Center 3x3 Grid)\n'
                 f'Total Tiles: {len(all_tile_coords)} ({tile_size}x{tile_size} size, {stride} stride)',
                 fontsize=14)
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Height (pixels)')

    # Set limits to zoom in, with 10% padding
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(max_y + padding_y, min_y - padding_y) # Inverted Y-axis

    ax.set_aspect('equal')
    ax.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # 8. Save the figure
    plot_filename = f"plot_tiling_strategy_CENTER_ZOOM_{scene_name}.png"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300)
    plt.close(fig)
    print(f"  - Saved center-zoomed tiling strategy plot to {output_filepath}")


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
    scientific_ndvi = raw_ndvi_array * scale_factor + offset
    scientific_ndvi_clipped = np.clip(scientific_ndvi, -1.0, 1.0)
    return (scientific_ndvi_clipped + 1.0) / 2.0

def normalize_hillshade(hillshade_array, max_val=255.0):
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


# --- Cloud Mask Selection Functions (Merged) ---

def get_cloud_cover(mtl_file_path):
    """
    Parses an .MTL.txt file and returns the CLOUD_COVER value.
    """
    try:
        with open(mtl_file_path, 'r') as f:
            for line in f:
                # Use regex to find the line and capture the value
                match = re.search(r'CLOUD_COVER\s*=\s*([0-9.]+)', line)
                if match:
                    # Found the line, return the value as a float
                    return float(match.group(1))
    except Exception as e:
        print(f"  [ERROR] Could not read or parse {mtl_file_path}: {e}")
    # Return None if not found or if an error occurred
    return None

def select_artificial_masks(metadata_folder, qa_files_folder, mode, settings):
    """
    Scans metadata, selects QA masks based on cloud cover, and returns a list of paths.
    """
    print("--- Starting Artificial Mask Selection ---")

    # --- 1. Set mode variables ---
    if mode == "DIVERSE":
        min_cloud = settings['DIVERSE_MIN_CLOUD']
        max_cloud = settings['DIVERSE_MAX_CLOUD']
        num_to_select = settings['DIVERSE_NUMBER_TO_SELECT']
        print(f"  - Mode: DIVERSE. Selecting {num_to_select} masks ({min_cloud}% - {max_cloud}%)")

    elif mode == "SPECIFIC":
        min_cloud = settings['SPECIFIC_MIN_CLOUD']
        max_cloud = settings['SPECIFIC_MAX_CLOUD']
        num_to_select = 1
        print(f"  - Mode: SPECIFIC. Selecting {num_to_select} mask ({min_cloud}% - {max_cloud}%)")

    else:
        print(f"  - ERROR: Invalid SELECTION_MODE: '{mode}'.")
        return []

    # --- 2. Validation ---
    if not metadata_folder.is_dir():
        print(f"  - ERROR: Metadata folder not found: {metadata_folder}")
        return []
    if not qa_files_folder.is_dir():
        print(f"  - ERROR: Source QA files folder not found: {qa_files_folder}")
        return []

    # --- 3. Find all candidates ---
    print(f"  - Scanning metadata in: {metadata_folder}")
    candidates = []
    for mtl_path in sorted(metadata_folder.glob("*_MTL.txt")):  # sorted -> reproducible with a fixed seed
        cloud_cover = get_cloud_cover(mtl_path)

        if cloud_cover is not None:
            if min_cloud <= cloud_cover <= max_cloud:
                base_id = mtl_path.name[:-8]
                qa_path = qa_files_folder / f"{base_id}_QA_PIXEL.TIF"
                if qa_path.exists():
                    candidates.append(qa_path)
                else:
                    print(f"    - [Warning] Found MTL for {base_id} but missing QA file: {qa_path.name}")

    print(f"  - Found {len(candidates)} total candidate masks matching criteria.")

    # --- 4. Select candidates ---
    if not candidates:
        print("  - ERROR: No candidate masks found. Cannot proceed.")
        return []

    if len(candidates) < num_to_select:
        print(f"  - Warning: Found only {len(candidates)} candidates, which is less than {num_to_select}.")
        print("  - Using all candidates found.")
        selected_paths = candidates
    else:
        selected_paths = random.sample(candidates, num_to_select)

    print(f"  - Successfully selected {len(selected_paths)} artificial masks.")
    print("--- End of Artificial Mask Selection ---")
    return selected_paths

# --- Adaptive Tiling and Inpainting Functions (Re-implemented) ---
# This logic is inspired by 'inpainting_lama_v3.py' but adapted
# for our specific model (21-ch float input) and data pipeline.

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
    - Skipped (Never Inpainted) - Only shown if present
    - Inpainted (Iteration 1, 2, 3...)
    """
    print(f"  - Visualizing final pixel category map for {scene_name}...")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 12 * (iteration_map.shape[0] / iteration_map.shape[1])))

    # --- 1. Dynamic Categories & Colors ---
    has_skipped = np.any(iteration_map == -1)
    max_iter_found = int(np.max(iteration_map))

    # Use 'magma' for perceptually uniform, sequential iteration mapping
    iter_cmap = plt.colormaps.get_cmap('YlGnBu_r').resampled(max(1, max_iter_found))

    color_list = []
    label_list = []
    bounds = []

    # Conditionally add 'Skipped' if -1 exists in the iteration map
    if has_skipped:
        color_list.append('#FF0000')  # Red
        label_list.append('Skipped (Hole)')
        bounds.extend([-1.5, -0.5])
    else:
        # Start bounds at -0.5 so Original Data aligns correctly
        bounds.append(-0.5)

    # Base color for Original Data
    color_list.append('#606060')  # Dark Gray
    label_list.append('Original Data')
    bounds.append(0.5)

    # Dynamically add colors/labels for iterations
    for i in range(1, max_iter_found + 1):
        # Calculate color index, preventing division by zero if max_iterations is 1
        cmap_idx = (i - 1) / max(1, max_iter_found - 1)
        color_list.append(iter_cmap(cmap_idx))
        label_list.append(f'Iteration {i}')
        bounds.append(i + 0.5)

    bounds = [float(b) for b in bounds]

    # Create the discrete colormap and normalizer
    cmap = ListedColormap(color_list)
    norm = BoundaryNorm(bounds, cmap.N)

    # --- 2. Plotting the Map ---
    im = ax.imshow(iteration_map, cmap=cmap, norm=norm)

    # Remove axis ticks, lines, and labels entirely
    ax.axis('off')

    # --- 3. Matching Colorbar Length to Scene ---
    # make_axes_locatable automatically sizes the colorbar axis to exactly match the image axis
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Calculate tick positions to be in the middle of each color
    tick_positions = [b + 0.5 for b in bounds[:-1]]
    cbar = fig.colorbar(im, cax=cax, ticks=tick_positions)
    cbar.ax.set_yticklabels(label_list, fontsize=16)

    #ax.set_title(f'Pixel Inpainting Category Map for {scene_name}', fontsize=16, pad=15)

    # Use bbox_inches='tight' to cleanly crop the removed axes margins
    plt.tight_layout()

    # Save the figure
    plot_filename = f"plot_inpainting_CATEGORY_MAP_{scene_name}.jpg"
    output_filepath = output_dir / plot_filename
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
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



# --- Main Processing Logic ---
if __name__ == '__main__':

    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate LaMa LST Inpainting")
    parser.add_argument('--config', type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()

    # 2. Load the configuration file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Reproducible choice of artificial masks (random.sample in select_artificial_masks)
    random.seed(config.get('random_seed', 42))

   # 3. Dynamically assign all paths from the config
    CHECKPOINT_PATH = config['checkpoint_path']
    MASK_LOOKUP_CSV_PATH = Path(config['mask_lookup_csv'])
    landsat_path = Path(config['landsat_dir_test'])
    LULC_Path = Path(config['lulc_dir'])
    DEM_Path = Path(config['dem_path'])
    SLOPE_Path = Path(config['slope_path'])
    ASPECT_Path = Path(config['aspect_path'])
    NDVI_Path = Path(config['ndvi_dir'])
    HILLSHADE_Path = Path(config['hillshade_dir'])
    SAR_Path = Path(config['sar_dir'])
    METADATA_FOLDER = Path(config['metadata_dir'])
    QA_FILES_FOLDER = Path(config['qa_files_dir'])

    output_dir_tiles = Path(config['output_dir_tiles'])
    output_dir_plots = Path(config['output_dir_plots'])
    output_dir_tiles.mkdir(parents=True, exist_ok=True)
    output_dir_plots.mkdir(parents=True, exist_ok=True)

    print(f"Loading mask lookup from {MASK_LOOKUP_CSV_PATH}...")
    mask_lookup = {} # Default to empty dictionary

    if MASK_LOOKUP_CSV_PATH.exists():
        try:
            # Read the input CSV
            df_metrics_lookup = pd.read_csv(MASK_LOOKUP_CSV_PATH)

            # Check if the required columns exist
            if 'scene_id' in df_metrics_lookup.columns and 'artificial_mask_id' in df_metrics_lookup.columns:
                # Create a dictionary: key=scene_id, value=artificial_mask_id
                mask_lookup = pd.Series(df_metrics_lookup.artificial_mask_id.values, index=df_metrics_lookup.scene_id).to_dict()
                print(f"  - Loaded {len(mask_lookup)} scene-to-mask mappings.")
            else:
                print(f"  - WARNING: {MASK_LOOKUP_CSV_PATH.name} is missing 'scene_id' or 'artificial_mask_id' columns.")
                print("  - Mask lookup will be empty.")
        except Exception as e:
            print(f"  - ERROR loading mask lookup from CSV: {e}")
            print("  - Mask lookup will be empty.")
    else:
        print(f"  - WARNING: Mask lookup file not found: {MASK_LOOKUP_CSV_PATH}")
        print("  - Mask lookup will be empty. (This is normal if not re-running old masks).")



    current_dir = os.getcwd()



    csv_filepath = output_dir_plots / "validation_metrics.csv"

    csv_headers = [
        'scene_id', 'missing_pixel_perc', 'validation_pixels', 'artificial_mask_id',
        'overall_mae', 'overall_rmse', 'overall_r',
        'q1_mae', 'q1_rmse', 'q2_mae', 'q2_rmse',
        'q3_mae', 'q3_rmse', 'q4_mae', 'q4_rmse'
    ]

    # Write header only if file doesn't exist
    if not csv_filepath.exists():
        with open(csv_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()


    # --- EFFICIENT REFACTOR ---
    # Define the cloud cover bins *once*
    CLOUD_COVER_BINS = {
        "low":    {"min": 30.0, "max": 35.0},
        "medium": {"min": 35.0, "max": 50.0},
        "high":   {"min": 50.0, "max": 70.0}
    }


    ACTIVE_INDICES = config.get('active_data_indices', list(range(20)))
    assert ACTIVE_INDICES[0] == 0, "active_data_indices must start with 0 (LST channel)"
    active_count = len(ACTIVE_INDICES)
    INPUT_CHANNELS = active_count + 1
    OUTPUT_CHANNELS = 1 # Output is always just the LST channel
    print(f"Loading LaMa model on {DEVICE} with {INPUT_CHANNELS} input channels...")
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
    print(f"Found {len(b10_files)} test scenes in {landsat_path}")
    qa_files = {p.name.split('_')[3]: p for p in find_landsat_files(landsat_path, "T1_QA_PIXEL.TIF")}


    for b10_path in b10_files:
        date_str = b10_path.name.split('_')[3]
        if date_str not in qa_files:
            print(f"  - Missing QA_PIXEL file for {date_str}, skipping.")
            continue
        qa_path = qa_files[date_str]
        print(f"--- Inpainting scene: {date_str} ---")

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
        binary_mask_ground_truth=(qa_bad_pixels | sensor_error_pixels).astype(np.uint8)

        doy_channel = np.full((ls_h, ls_w), acquisition_date.timetuple().tm_yday, dtype=np.uint16)

        ground_truth = convert_b10toKelvin(ls_data)
        ground_truth[binary_mask_ground_truth==1]=0

        stack_ground_truth=np.stack([ground_truth,
                        binary_mask_ground_truth])


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

        for bin_name, bin_range in CLOUD_COVER_BINS.items():

            print(f"\n  ==> Processing bin '{bin_name}' ({bin_range['min']}-{bin_range['max']}%) for scene {date_str} <==")

            # --- Unique run ID ---
            run_id = f"{date_str}_{bin_name}"

            # --- Check for specific mask from lookup CSV ---
            artificial_mask_name = None
            selected_mask_path = None

            if run_id in mask_lookup:
                # 1. We found this run_id in the old CSV
                specific_mask_name = mask_lookup[run_id]
                specific_mask_path = QA_FILES_FOLDER / specific_mask_name

                if specific_mask_path.exists():
                    print(f"  - Found '{run_id}' in lookup. Using specific mask: {specific_mask_name}")
                    selected_mask_path = specific_mask_path
                    artificial_mask_name = specific_mask_name
                else:
                    print(f"  - WARNING: Looked up mask '{specific_mask_name}' for '{run_id}' but it does NOT exist.")
                    print(f"  - Path checked: {specific_mask_path}")
                    print("  - Skipping this run_id.")
                    continue # Skip this bin

            else:
                # 2. run_id not in lookup, so select one based on cloud cover (original logic)
                print(f"  - '{run_id}' not in lookup. Selecting new mask for bin '{bin_name}'.")
                current_mask_settings = {
                    "SPECIFIC_MIN_CLOUD": bin_range['min'],
                    "SPECIFIC_MAX_CLOUD": bin_range['max']
                }
                selected_mask_paths = select_artificial_masks(
                   METADATA_FOLDER,
                   QA_FILES_FOLDER,
                   "SPECIFIC",
                   current_mask_settings
                )

                if not selected_mask_paths:
                    print(f"  - WARNING: No new artificial masks found for bin '{bin_name}'. Skipping this run.")
                    continue

                selected_mask_path = selected_mask_paths[0]
                artificial_mask_name = selected_mask_path.name

            # --- END of new logic block ---


            qa_data_artificial_reproj = np.zeros((ls_h, ls_w), dtype=qa_data.dtype)
            with rio.open(selected_mask_path) as art_src:
                art_profile = art_src.profile
                art_data = art_src.read(1)
                reproject(
                    source=art_data, destination=qa_data_artificial_reproj,
                    src_transform=art_profile['transform'], src_crs=art_profile['crs'],
                    dst_transform=ls_trans, dst_crs=ls_crs,
                    resampling=Resampling.nearest)

            artificial_qa_bits = qa_masks(qa_data_artificial_reproj)
            artificial_bad_mask = (artificial_qa_bits['cloud'] | artificial_qa_bits['shadow'] |
                                   artificial_qa_bits['cirrus'] | artificial_qa_bits['dilated'] |
                                   artificial_qa_bits['snow'])
            print(f"  - Artificial mask adds {np.sum(artificial_bad_mask)} pixels.")


            binary_mask = (binary_mask_ground_truth.astype(bool) | artificial_bad_mask).astype(np.uint8)
            print(f"  - Total pixels to mask (original + artificial): {np.sum(binary_mask)}")


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

            unrotated_stack, _ = crop_and_unrotate(stack, merged_poly, angle, ls_trans)
            unrotated_stack_groud_truth, _ = crop_and_unrotate(stack_ground_truth, merged_poly, angle, ls_trans)
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

            ## commenteed to get rid of the error appearing afte adapting the adaptive inpainting function
            # unnormalized_data = trimmed_stack[:-1]
            # final_binary_mask = np.nan_to_num(trimmed_stack[-1]).astype(np.uint8)


            # Extract the Ground Truth LST and mask it based on the UNIFIED trimmed stack.
            gt = trimmed_stack_gt[0].copy()
            # The mask here is the original scene mask (channel 1 of the GT stack)
            gt[trimmed_stack_gt[1] == 1] = np.nan

            ## commenteed to get rid of the error appearing afte adapting the adaptive inpainting function
            # gt = trimmed_stack_gt[0].copy()
            # # 2. Apply the mask (where mask is 1, set LST to nan)
            # gt[trimmed_stack_gt[1] == 1] = np.nan


            unique_vals = np.unique(final_binary_mask)
            print(f"  - DEBUG: Unique values in final_binary_mask before saving: {unique_vals}")


            # --- Save the 'final_binary_mask' as TIFF ---
            mask_profile = {
               'driver': 'GTiff',
               'height': final_binary_mask.shape[0],
               'width': final_binary_mask.shape[1],
               'count': 1,
               'dtype': 'uint8', # Should be uint8 based on your script
               'compress': 'lzw',
               # NO CRS or TRANSFORM - Georeferencing is lost due to unrotate/trim
            }

            mask_save_path = output_dir_tiles / f"{run_id}_missing_data_mask.tif"
            with rio.open(mask_save_path, 'w', **mask_profile) as dst:
                dst.write(final_binary_mask, 1)
            print(f"  - Saved rotated/trimmed missing data mask to: {mask_save_path.name}")
            # ---> END OF NEW BLOCK <---

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



            #--- Save the 'ground_truth' array for validation (as .npy) ---
            gt_save_path = output_dir_tiles / f"{run_id}_ground_truth2.tif"
            with rio.open(gt_save_path, 'w', **gt_profile) as dst:
                dst.write(gt.astype(np.float32), 1)



            norm_cfg = config['normalization']
            normalized_lst = normalize_lst(unnormalized_data[0],norm_cfg['lst']['min'],norm_cfg['lst']['max'])
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


            # --- ADAPTIVE TILE AND INPAINT ---
            # Feed the model only the channels it was trained with (config: active_data_indices).
            # LST must stay at index 0 because adaptive_inpainting() updates channel 0.
            model_stack = normalized_stack[ACTIVE_INDICES]
            if model:
                # Pass the 20-channel data stack and the 1-channel (squeezed) mask
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



                # --- ADDED: Call the new category plot function ---
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

                    # --- Composite final image ---
                    # We need the original LST (unmasked) for this
                    # Original LST, clipped to the normalisation range exactly like the model input
                    # (the original call normalize_lst(x) lacked min/max and raised a TypeError)
                    original_lst_kelvin = normalize_lst(unnormalized_data[0], min_k, max_k)
                    original_lst_kelvin = (original_lst_kelvin * (max_k - min_k)) + min_k

                    # Where the mask was 1 (a hole), use inpainted. Where 0, use original.
                    final_composite_kelvin = np.where(
                        final_binary_mask.astype(bool),   # The rotated/trimmed boolean mask (True=hole)
                        inpainted_lst_kelvin,       # Use inpainted value
                        original_lst_kelvin         # Use original value
                    )



                    # Create metadata for the TIFF file
                    # Create basic metadata for the TIFF file
                    final_output_profile = {
                        'driver': 'GTiff',
                        'height': final_composite_kelvin.shape[0], # Correct index for height
                        'width': final_composite_kelvin.shape[1],  # Correct index for width
                        'count': 1,
                        'dtype': final_composite_kelvin.dtype,
                        'compress': 'lzw',
                        # NO CRS or TRANSFORM - Georeferencing is lost
                        'nodata': np.nan
                    }


                    tiff_save_path = output_dir_tiles / f"{run_id}_inpainted_final_256.tif"
                    print(f"  - Saving final inpainted GeoTIFF to: {tiff_save_path.name}")
                    with rio.open(tiff_save_path, 'w', **final_output_profile) as dst:
                        dst.write(final_composite_kelvin.astype(np.float32), 1)


                    #np.save(gt_save_path, ground_truth_final)

                    # --- Save the 'inpainted' array (full size) for validation (as .npy) ---
                    inpainted_save_path = output_dir_tiles / f"{run_id}_inpainted_values_256.npy"  # was date_str -> overwritten per bin
                    np.save(inpainted_save_path, inpainted_lst_kelvin)

                    print(f"  - Saved rotated/trimmed ground truth for validation: {gt_save_path.name}")
                    print(f"  - Saved rotated/trimmed inpainted values for validation: {tiff_save_path.name}")


                    pixel_error_map = np.abs(inpainted_lst_kelvin - gt)

                    error_save_path = output_dir_tiles / f"{run_id}_pixel_error2.tif"
                    print(f"  - Saving pixel-wise error map to: {error_save_path.name}")
                    with rio.open(error_save_path, 'w', **final_output_profile) as dst:
                        dst.write(pixel_error_map.astype(np.float32), 1)

                    # --- Calculate Validation Metrics ---
                    metrics = calculate_and_print_metrics(
                       gt,
                       inpainted_lst_kelvin,
                       final_binary_mask,
                       run_id,
                       output_dir_plots
                    )

                    # 2. Check if metrics were successfully calculated
                    if metrics:
                        # 3. Calculate total missing pixel percentage
                        missing_perc = (np.sum(final_binary_mask) / final_binary_mask.size) * 100

                        # 4. Create the full data row
                        data_row = {
                            "scene_id": run_id,
                            "missing_pixel_perc": missing_perc,
                            "artificial_mask_id": artificial_mask_name,
                            **metrics # Merge the returned metrics dict
                        }

                        # 5. Append the row to the CSV
                        try:
                            with open(csv_filepath, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=csv_headers)
                                writer.writerow(data_row)
                            print(f"  - Metrics successfully saved to {csv_filepath.name}")
                        except Exception as e:
                            print(f"  - ERROR: Could not write metrics to CSV: {e}")

                    else:
                        print("  - No metrics returned, skipping CSV write.")

                else:
                    print("  - ERROR: Tiling and inpainting failed.")

            gc.collect() # Clean up memory

    print("\n--- All scenes processed. ---")



