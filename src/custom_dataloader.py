import torch
from torch.utils.data import Dataset
import rasterio
from PIL import Image
import numpy as np
from pathlib import Path
import random

class LSTInpaintingDataset(Dataset):
    """
    Custom PyTorch Dataset for loading LST training tiles and applying random masks.
    """
    def __init__(self, training_tile_dir, mask_tile_dir,active_indices=None, tile_list=None):
        """
        Args:
            training_tile_dir (str or Path): Directory containing the clean, multi-channel TIFF tiles.
            mask_tile_dir (str or Path): Directory containing the single-channel PNG mask tiles.
        """
        
        if tile_list:
           # For previews, use the specific list of tile paths provided
           self.training_tile_paths = [Path(p) for p in tile_list]
        else:
           # For regular training/validation, find all .tif files
           self.training_tile_paths = sorted(list(Path(training_tile_dir).glob('*.tif')))
        self.mask_tile_paths = sorted(list(Path(mask_tile_dir).glob('*.png')))
                                                                   
        # self.training_tile_paths = sorted(list(Path(training_tile_dir).glob('*.tif')))
        # self.mask_tile_paths = sorted(list(Path(mask_tile_dir).glob('*.png')))
        
        if not self.training_tile_paths:
            raise IOError(f"No training tiles found in {training_tile_dir}")
        if not self.mask_tile_paths:
            raise IOError(f"No mask tiles found in {mask_tile_dir}")
            
        print(f"Found {len(self.training_tile_paths)} training tiles.")
        print(f"Found {len(self.mask_tile_paths)} mask tiles.")

        ## ----------------------------------------------------------------
        ## --- Automatically detect channel count ---
        # Check the number of channels from the first tile to make the script flexible
        with rasterio.open(self.training_tile_paths[0]) as src:
            # We assume the last channel in the file is a placeholder mask from the
            # generation script, so we subtract 1 to get the number of data channels.
            self.num_data_channels = src.count - 1
        print(f"Detected {self.num_data_channels} data channels per tile.")
        ## ----------------------------------------------------------------
        if active_indices is None:
            self.active_indices = list(range(12)) 
            print(f"No indices provided. Defaulting to Stage 1 (12 data channels).")
        else:
            self.active_indices = active_indices
            print(f"Using custom active indices: {self.active_indices}")

    def __len__(self):
        """
        Returns the total number of training tiles.
        """
        return len(self.training_tile_paths)

    def __getitem__(self, idx):
        """
        Fetches one training sample and applies a random mask.
        """
        # 1. Load a clean training tile
        with rasterio.open(self.training_tile_paths[idx]) as src:
            clean_tile_data = src.read()
        
        ## ----------------------------------------------------------------
        ## --- Use all available data channels ---
        # Instead of hardcoding 12, use the detected number of data channels.
        #########data_channels = clean_tile_data[:self.num_data_channels, :, :].astype(np.float32)
        data_channels = clean_tile_data[self.active_indices, :, :].astype(np.float32)
        ## ----------------------------------------------------------------
    

        # 2. Create the Ground Truth Tensor FIRST.
        # This creates a completely independent copy of the LST data (channel 0) before any masking happens.
        ground_truth_lst_tensor = torch.from_numpy(data_channels[0:1, :, :]).clone()
        
        # 3. Load a random mask tile (this logic is unchanged)
        random_mask_path = random.choice(self.mask_tile_paths)
        mask_image = Image.open(random_mask_path).convert('L')
        mask_array = np.array(mask_image) / 255.0
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        # 4. Create the model input tensor from the full set of data channels
        model_input_data = torch.from_numpy(data_channels)
        
        # 5. Mask only the first channel (LST) (this logic is unchanged)
        model_input_data[0, :, :] *= (1 - mask_tensor.squeeze(0))

        # 6. Concatenate the data channels and the mask to create the final model input
        # This will now correctly be (e.g.) 17 data channels + 1 mask channel = 18 total
        final_model_input = torch.cat([model_input_data, mask_tensor], dim=0)
        
        tile_name = self.training_tile_paths[idx].name
        return {
            'image': final_model_input,
            'gt_image': ground_truth_lst_tensor,
            'tile_name': tile_name
        }
    
