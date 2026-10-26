"""
Exports, for every Landsat 8/9 scene of a WRS-2 path/row, a Sentinel-1 (VV, VH)
median "stitch" on the Landsat 30 m grid to Google Drive using Google Earth Engine.
A 15-day window centred on the Landsat date is used when it fully covers the
scene, otherwise a 50-day window.

Requires an Earth Engine account and a Cloud project:
    pip install earthengine-api
    earthengine authenticate

Usage:
    python src/downloading_sar_data.py --project YOUR_GEE_PROJECT \
        --path 193 --row 26 --start 2013-04-01 --end 2025-06-01
The GeoTIFFs appear in the Drive folder given by --drive-folder; download them
into `paths.sar_dir` of configs/data_config.yaml.
"""
import argparse
import sys

import ee

# --- Defaults (overridable from the command line) ---
LANDSAT_CLOUD_THRESHOLD = 81      # Cloud cover < 81%
S1_ORBIT_PASS = 'DESCENDING'      # Use DESCENDING orbit
COVERAGE_THRESHOLD = 1.0          # 100% coverage required for the 15-day window
EXPORT_FOLDER = 'GEE_S1_Stitch_Exports'  # Folder in your Google Drive

# S1 Window sizes
S1_HALF_WINDOW_1 = 7.5  # 15-day total window
S1_HALF_WINDOW_2 = 25   # 50-day total window

# --- S1 Processing Functions ---

def crop_s1_swaths(image):
    """
    Masks S1 edges by removing pixels outside the reliable incidence angle range.
    This is a geometric crop, not a data-value threshold.
    """
    angle = image.select('angle')
    mask = angle.gte(31).And(angle.lte(45))
    return image.updateMask(mask)

def check_coverage(collection, geom):
    """
    Checks the fractional coverage of a mosaic within a given geometry.
    """
    collectionSize = collection.size()
    
    return ee.Algorithms.If(
        collectionSize.eq(0),
        ee.Number(0),
        collection.mosaic().clip(geom).select('VV').mask()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=1000, 
                maxPixels=1e9
            ).get('VV')
    )

# --- Main Function to Process ONE Landsat Scene ---

def create_s1_stitch(landsat_scene):
    """
    For a single Landsat scene, generates the corresponding S1 median stitch.
    """
    # Cast to ee.Image to help with property access
    landsat_scene = ee.Image(landsat_scene)
    
    # Get date, geometry, and now CRS from the Landsat scene
    l_date = landsat_scene.date()
    l_geom = landsat_scene.geometry()
    l_id = landsat_scene.get('system:id')
    
    # --- Get the CRS from a Landsat band ---
    # We use a reflective band (e.g., SR_B2) to get the 30m projection
    l_crs = landsat_scene.select('SR_B2').projection().crs()

    # 2. Define S1 Search Windows
    startDate15 = l_date.advance(-S1_HALF_WINDOW_1, 'day')
    endDate15 = l_date.advance(S1_HALF_WINDOW_1, 'day')
    startDate50 = l_date.advance(-S1_HALF_WINDOW_2, 'day') 
    endDate50 = l_date.advance(S1_HALF_WINDOW_2, 'day')

    # 3. Base Sentinel-1 Collection
    s1Base = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.eq('orbitProperties_pass', S1_ORBIT_PASS)) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filterBounds(l_geom) \
        .map(crop_s1_swaths) \
        .select(['VV', 'VH']) 

    # 4. Create time-windowed S1 Collections
    s1_15day = s1Base.filterDate(startDate15, endDate15)
    s1_50day = s1Base.filterDate(startDate50, endDate50)

    # 6. Calculate 15-day Coverage and Create Median Stitches
    coverage15 = ee.Number(check_coverage(s1_15day, l_geom))
    median15 = s1_15day.median().clip(l_geom)
    median50 = s1_50day.median().clip(l_geom) 

    # 7. Conditionally Select the Final Median Stitch
    use15day = coverage15.gte(COVERAGE_THRESHOLD)
    finalStitch = ee.Image(ee.Algorithms.If(
        use15day,
        median15,
        median50 
    ))
    
    # 8. Determine status
    status = ee.Algorithms.If(
        use15day,
        'Used_15-day_window',
        'Used_50-day_window'
    )

    # Return the final S1 image, now with the Landsat CRS stored
    return finalStitch.set({
        'landsat_id': l_id,
        'landsat_date': l_date.format('YYYY-MM-dd'),
        'processing_status': status,
        'coverage_15day': coverage15,
        'landsat_crs': l_crs  # --- Store the CRS ---
    })

def main():
    ap = argparse.ArgumentParser(description="Export Sentinel-1 stitches matching Landsat scenes (GEE)")
    ap.add_argument("--project", required=True, help="Your Google Earth Engine / Cloud project ID")
    ap.add_argument("--path", type=int, default=193, help="Landsat WRS-2 path")
    ap.add_argument("--row", type=int, default=26, help="Landsat WRS-2 row")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD (exclusive)")
    ap.add_argument("--max-cloud", type=float, default=LANDSAT_CLOUD_THRESHOLD)
    ap.add_argument("--drive-folder", default=EXPORT_FOLDER)
    args = ap.parse_args()

    try:
        ee.Initialize(project=args.project)
        print("Google Earth Engine has been initialized successfully.")
    except Exception as e:
        print(f"Error initializing GEE: {e}")
        print("Please run `earthengine authenticate` and pass a valid --project.")
        sys.exit(1)

    # --- 1. Find Qualifying Landsat Scenes ---
    landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    landsat9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')

    filteredLandsat = landsat8.merge(landsat9) \
        .filter(ee.Filter.eq('WRS_PATH', args.path)) \
        .filter(ee.Filter.eq('WRS_ROW', args.row)) \
        .filterDate(args.start, args.end) \
        .filter(ee.Filter.lt('CLOUD_COVER', args.max_cloud)) \
        .sort('system:time_start')

    sceneCount = filteredLandsat.size().getInfo()
    print(f'Landsat scenes found: {sceneCount}')

    # --- 2. Generate S1 Stitches for EACH Landsat Scene ---
    s1_Stitches = filteredLandsat.map(create_s1_stitch)
    print(f'Generated S1 Stitch Collection: {s1_Stitches.size().getInfo()} elements')

    # --- 3. Batch Export to Drive ---
    print('--- Export ---')
    image_list = s1_Stitches.toList(s1_Stitches.size())
    landsat_ids = s1_Stitches.aggregate_array('landsat_id')
    landsat_dates = s1_Stitches.aggregate_array('landsat_date')
    landsat_crs_list = s1_Stitches.aggregate_array('landsat_crs')

    # Fetch the properties to the client: [[id, date], crs]
    properties_list = landsat_ids.zip(landsat_dates).zip(landsat_crs_list).getInfo()
    print(f'Preparing and starting {len(properties_list)} export tasks...')

    for index, props_combined in enumerate(properties_list):
        props_part1, landsat_crs = props_combined
        landsat_id, landsat_date = props_part1

        s1_image = ee.Image(image_list.get(index))
        geom = s1_image.geometry()

        # File name pattern expected by the tiling scripts:
        # S1_Stitch_<YYYY-MM-DD>_<SENSOR>_<PATHROW>_<YYYYMMDD>.tif
        landsat_name = landsat_id.split('/')[-1]
        description = f'S1_Stitch_for_{landsat_name}'
        file_name = f'S1_Stitch_{landsat_date}_{landsat_name}'

        task = ee.batch.Export.image.toDrive(
            image=s1_image.select(['VV', 'VH']).toFloat(),
            description=description,
            fileNamePrefix=file_name,
            folder=args.drive_folder,
            region=geom.bounds(),
            crs=landsat_crs,   # match the Landsat grid
            scale=30,
            maxPixels=1e13
        )
        task.start()

        if (index + 1) % 10 == 0 or (index + 1) == len(properties_list):
            print(f'Started task {index + 1}/{len(properties_list)}: {description}')

    print('All export tasks started. Monitor them in the GEE Code Editor Task tab or with `ee.batch.Task.list()`.')


if __name__ == '__main__':
    main()
