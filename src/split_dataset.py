"""
Splits the generated training tiles into train / val / test folders (80/10/10).

NOTE: files are MOVED (not copied) to save disk space, so the source folder
(`paths.tiles_out`) is empty afterwards.

Usage (from the repository root):
    python src/split_dataset.py --config configs/data_config.yaml
"""
import argparse
import random
import shutil
from pathlib import Path

import yaml

def split_dataset(tile_dir, output_dir, train_split=0.8, val_split=0.1, seed=42):
    """
    Splits the clean training tiles into train, validation, and test sets.
    This script NO LONGER copies the masks to save disk space.
    """
    # --- Setup Directories ---
    tile_dir = Path(tile_dir)
    output_dir = Path(output_dir)

    train_tiles_path = output_dir / "train" / "tiles"
    val_tiles_path = output_dir / "val" / "tiles"
    test_tiles_path = output_dir / "test" / "tiles"

    # Create all necessary directories
    for path in [train_tiles_path, val_tiles_path, test_tiles_path]:
        path.mkdir(parents=True, exist_ok=True)

    # --- Get and Shuffle File Lists ---
    all_tile_files = sorted(list(tile_dir.glob("*.tif")))
    if not all_tile_files:
        raise FileNotFoundError(f"No .tif tiles found in {tile_dir}")
    
    random.Random(seed).shuffle(all_tile_files)  # seeded for reproducibility
    
    # --- Split Tiles ---
    num_tiles = len(all_tile_files)
    train_end = int(num_tiles * train_split)
    val_end = train_end + int(num_tiles * val_split)

    train_set = all_tile_files[:train_end]
    val_set = all_tile_files[train_end:val_end]
    test_set = all_tile_files[val_end:]

    def move_files(file_list, dest_dir):
        print(f"moving {len(file_list)} files to {dest_dir}...")
        for f in file_list:
            shutil.move(str(f), str(dest_dir / f.name))

    move_files(train_set, train_tiles_path)
    move_files(val_set, val_tiles_path)
    move_files(test_set, test_tiles_path)
    
    print(f"\nDataset split complete!")
    print(f"Total tiles: {num_tiles}")
    print(f"  - Training set: {len(train_set)} tiles")
    print(f"  - Validation set: {len(val_set)} tiles")
    print(f"  - Test set: {len(test_set)} tiles")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split training tiles into train/val/test")
    parser.add_argument('--config', type=str, default='configs/data_config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    SOURCE_TILES_DIR = Path(config['paths']['tiles_out'])
    SPLIT_DATA_DIR = Path(config['paths']['split_data_dir'])
    seed = config.get('parameters', {}).get('random_seed', 42)

    split_dataset(SOURCE_TILES_DIR, SPLIT_DATA_DIR, seed=seed)

