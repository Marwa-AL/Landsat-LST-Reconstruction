"""
Automates the generation of hillshade rasters for Landsat scenes.

This script reads sun position and corner coordinates from Landsat metadata
files (*_MTL.txt), crops a master DEM to the scene's extent, and then
generates the scene-specific hillshade.


All paths and the target CRS are read from configs/data_config.yaml.

Usage (from the repository root):
    python src/generate_hillshades.py --config configs/data_config.yaml

NOTE: the Sobel gradients in generate_numpy_hillshade() are not divided by
(8 * pixel size), so slopes are strongly exaggerated compared with a standard
hillshade (e.g. `gdaldem hillshade`). The published dataset was produced with
this exact formula, so it is kept unchanged here for reproducibility.
"""
import argparse
import yaml
from pathlib import Path
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from scipy.ndimage import convolve

def generate_numpy_hillshade(dem_array, elevation, azimuth, nodata_value):
    """
    Generates a hillshade from a DEM array using numpy and scipy.
    This function is a self-contained replacement for the richdem dependency.
    """
    # Define Sobel filters to calculate slope and aspect
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    # Handle nodata values by creating a mask
    valid_data_mask = dem_array != nodata_value
    
    # Calculate gradients
    x_grad = convolve(dem_array, kernel_x)
    y_grad = convolve(dem_array, kernel_y)
    
    # Calculate slope and aspect in radians
    slope_rad = np.arctan(np.sqrt(x_grad**2 + y_grad**2))
    aspect_rad = np.arctan2(y_grad, -x_grad) # Aspect measured from North

    # Convert sun position from degrees to radians
    azimuth_rad = np.deg2rad(azimuth)
    zenith_rad = np.deg2rad(90 - elevation)
    
    # Hillshade formula
    hillshade = 255 * (
        (np.cos(zenith_rad) * np.cos(slope_rad)) +
        (np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad))
    )
    
    # Clamp values to 0-255 and apply the nodata mask back
    hillshade = np.clip(hillshade, 0, 255)
    hillshade[~valid_data_mask] = 0 # Set nodata areas in hillshade to 0
    
    return hillshade.astype(np.uint8)

def parse_mtl_file(mtl_path, keys_to_find):
    """
    Parses a Landsat MTL file for a specific set of keys.
    """
    data = {}
    with open(mtl_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(' = ')]
            if len(parts) == 2:
                key, value = parts
                if any(key.endswith(req_key) for req_key in keys_to_find):
                    actual_key = next((req for req in keys_to_find if key.endswith(req)), None)
                    try:
                        if 'ZONE' in actual_key: data[actual_key] = int(value)
                        else: data[actual_key] = float(value)
                    except (ValueError, TypeError): data[actual_key] = value

    if not all(key in data for key in keys_to_find):
        missing = [key for key in keys_to_find if key not in data]
        raise ValueError(f"Could not find: {', '.join(missing)}")
    return data

def reproject_dem_to_crs(source_dem_path, target_dem_path, target_crs):
    """Reprojects a DEM to a new CRS and saves it."""
    print(f"  - Reprojecting master DEM to {target_crs.to_string()}...")
    with rasterio.open(source_dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds)
        
        meta = src.meta.copy()
        meta.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'compress': 'lzw'
        })

        with rasterio.open(target_dem_path, 'w', **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear)
    print(f"  - Saved reprojected DEM to: {target_dem_path.name}")

def create_cropped_hillshade(dem_path, output_path, scene_info, target_crs):
    """
    Crops a pre-reprojected DEM and generates a hillshade.
    """
    azimuth = scene_info['SUN_AZIMUTH']
    elevation = scene_info['SUN_ELEVATION']
    
    print(f"  - Generating hillshade with Azimuth: {azimuth:.2f}, Elevation: {elevation:.2f}")

    geom = {
        'type': 'Polygon',
        'coordinates': [[
            [scene_info['CORNER_UL_PROJECTION_X_PRODUCT'], scene_info['CORNER_UL_PROJECTION_Y_PRODUCT']],
            [scene_info['CORNER_UR_PROJECTION_X_PRODUCT'], scene_info['CORNER_UR_PROJECTION_Y_PRODUCT']],
            [scene_info['CORNER_LR_PROJECTION_X_PRODUCT'], scene_info['CORNER_LR_PROJECTION_Y_PRODUCT']],
            [scene_info['CORNER_LL_PROJECTION_X_PRODUCT'], scene_info['CORNER_LL_PROJECTION_Y_PRODUCT']],
            [scene_info['CORNER_UL_PROJECTION_X_PRODUCT'], scene_info['CORNER_UL_PROJECTION_Y_PRODUCT']]
        ]]
    }

    with rasterio.open(dem_path) as src:
        source_nodata = src.nodata
        # target_crs is already a rasterio CRS object (see main()).
        if src.crs != target_crs:
            raise ValueError(f"CRS mismatch in create_cropped_hillshade. DEM is {src.crs}, expected {target_crs}.")

        cropped_dem_array, cropped_transform = mask(
            src, 
            [geom], 
            crop=True, 
            all_touched=True,
            nodata=source_nodata
        )
        meta = src.meta.copy()

    meta.update({
        "driver": "GTiff",
        "height": cropped_dem_array.shape[1],
        "width": cropped_dem_array.shape[2],
        "transform": cropped_transform
    })
    
    dem_squeezed = cropped_dem_array.squeeze()
    
    if source_nodata is None:
        nodata_value = -9999
        print(f"  - WARNING: Source DEM has no 'nodata' value set. Using {nodata_value} as a fallback.")
        dem_squeezed[np.isnan(dem_squeezed)] = nodata_value
    else:
        nodata_value = source_nodata

    # Call the new self-contained hillshade function
    hillshade_array = generate_numpy_hillshade(dem_squeezed, elevation, azimuth, nodata_value)
    
    meta.update(dtype='uint8', count=1, nodata=0, compress='lzw')

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(hillshade_array, 1)
    print(f"  - Successfully saved cropped hillshade to: {output_path.name}")

def main():
    
    """Main function to orchestrate the hillshade generation."""
    parser = argparse.ArgumentParser(description="Generate scene-specific hillshades from the DEM")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    metadata_dir = Path(config['paths']['metadata'])
    dem_path =Path(config['paths']['dem_path'])
    output_dir = Path(config['paths']['hillshade_dir'])

    target_crs = CRS.from_epsg(config['parameters']['target_crs_epsg'])

    if not all([metadata_dir.is_dir(), dem_path.is_file()]):
        print("Error: Please ensure metadata directory and DEM file paths are correct.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    mtl_files = list(metadata_dir.glob('*_MTL.txt'))
    print(f"Found {len(mtl_files)} metadata files.")

    # --- STAGE 1: Reproject master DEM to target CRS if necessary ---
    print(f"\n--- STAGE 1: Preparing DEM in target CRS ({target_crs.to_string()}) ---")
    with rasterio.open(dem_path) as src:
        if src.crs == target_crs:
            print("  - Master DEM is already in the target CRS. No reprojection needed.")
            dem_to_use_for_all_scenes = dem_path
        else:
            print(f"  - Master DEM CRS ({src.crs.to_string()}) does not match target CRS.")
            epsg_code = config['parameters']['target_crs_epsg'] 
            reprojected_dem_path = output_dir / f"DEM_reprojected_EPSG{epsg_code}.tif"
            if not reprojected_dem_path.exists():
                reproject_dem_to_crs(dem_path, reprojected_dem_path, target_crs)
            else:
                print(f"  - Using existing reprojected DEM: {reprojected_dem_path.name}")
            dem_to_use_for_all_scenes = reprojected_dem_path

    # --- STAGE 2: Process each scene using the reprojected DEM ---
    print(f"\n--- STAGE 2: Generating hillshades for each scene ---")
    print("-" * 50)
    for mtl_file in mtl_files:
        print(f"Processing: {mtl_file.name}")
        try:
            full_keys = [
                'SUN_AZIMUTH', 'SUN_ELEVATION', 'LANDSAT_PRODUCT_ID',
                'CORNER_UL_PROJECTION_X_PRODUCT', 'CORNER_UL_PROJECTION_Y_PRODUCT',
                'CORNER_UR_PROJECTION_X_PRODUCT', 'CORNER_UR_PROJECTION_Y_PRODUCT',
                'CORNER_LL_PROJECTION_X_PRODUCT', 'CORNER_LL_PROJECTION_Y_PRODUCT',
                'CORNER_LR_PROJECTION_X_PRODUCT', 'CORNER_LR_PROJECTION_Y_PRODUCT'
            ]
            scene_info = parse_mtl_file(mtl_file, full_keys)
            
            output_filename = f"{scene_info['LANDSAT_PRODUCT_ID']}_hillshade.tif"
            output_path = output_dir / output_filename

            if output_path.exists():
                print("  - Hillshade file already exists. Skipping.")
                continue
            
            create_cropped_hillshade(dem_to_use_for_all_scenes, output_path, scene_info, target_crs)

        except (ValueError, KeyError) as e:
            print(f"  - SKIPPING: Could not process file. Reason: {e}")
        except Exception as e:
            print(f"  - SKIPPING: An unexpected error occurred. Reason: {e}")
        finally:
            print("-" * 50)

    print("Processing complete.")

if __name__ == '__main__':
    main()

