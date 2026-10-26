"""
Trains the multi-modal LaMa (FFC) generator + NLayer PatchGAN discriminator for
LST gap filling.

Usage (from the repository root):
    python src/train.py --config configs/config.yaml
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless backend (servers / clusters)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips

from models import build_generator
from custom_dataloader import LSTInpaintingDataset



# -----------------------------------------------------------------------------
# MODEL AND LOSS DEFINITIONS (kept from original for simplicity)
# -----------------------------------------------------------------------------

def log_cosh_loss(y_pred, y_true):
    return torch.log(torch.cosh(y_pred - y_true)).mean()

def save_model_architecture(model, filepath):
    """Saves the architecture of a PyTorch model to a text file."""
    try:
        with open(filepath, 'w') as f:
            f.write(str(model))
        print(f"Model architecture saved to {filepath}")
    except Exception as e:
        print(f"Failed to save model architecture to {filepath}. Error: {e}")


def generate_validation_plots(writer, generator, dataloader, device, epoch, output_dir, config, num_plots=2):
    """
    Generates validation plots, now including a histogram of the error distribution.
    """
    generator.eval()
    plot_dir = output_dir / "validation_previews"
    plot_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        batch = next(iter(dataloader))
        model_input = batch['image'].to(device)
        ground_truth = batch['gt_image'].to(device)
        tile_names = batch['tile_name']
        mask = model_input[:, -1:, :, :] # Assuming mask is the last channel

        generated_output = generator(model_input)
        composite_output = ground_truth * (1 - mask) + generated_output * mask

        for i in range(min(num_plots, model_input.size(0))):
            generated = composite_output[i].detach().cpu().numpy().squeeze()
            gt = ground_truth[i].cpu().numpy().squeeze()
            current_tile_name = Path(tile_names[i]).stem

            def unnormalize(img_norm):
                min_k=config['normalization']['lst']['min']
                max_k=config['normalization']['lst']['max']
                return img_norm * (max_k - min_k) + min_k

            generated_kelvin = unnormalize(generated)
            gt_kelvin = unnormalize(gt)

            input_mask = mask[i].cpu().numpy().squeeze().astype(bool)
            masked_input_for_plot = np.ma.masked_where(input_mask, gt_kelvin)

            difference_map = np.abs(generated_kelvin - gt_kelvin)
            difference_map[~input_mask] = 0

            # --- Extract errors from only the masked region for the histogram ---
            errors_in_mask = difference_map[input_mask]

            vmin = np.nanmin(gt_kelvin)
            vmax = np.nanmax(gt_kelvin)

            # --- Change plot layout from (1, 4) to (1, 5) ---
            fig, axes = plt.subplots(1, 5, figsize=(30, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.8]})
            fig.suptitle(f"Validation Preview - Tile: {current_tile_name} - Epoch {epoch+1}", fontsize=16)

            cmap = plt.cm.viridis
            cmap.set_bad(color='black')

            axes[0].imshow(masked_input_for_plot, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            axes[0].set_title("Masked Input")
            axes[0].axis("off")

            im = axes[1].imshow(generated_kelvin, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            axes[1].set_title("Inpainted Output")
            axes[1].axis("off")

            axes[2].imshow(gt_kelvin, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

            im_diff = axes[3].imshow(difference_map, cmap='hot')
            axes[3].set_title("Absolute Difference (Error) [K]")
            axes[3].axis("off")

            # --- Add the new histogram plot to the 5th panel ---
            axes[4].hist(errors_in_mask, bins=50, color='crimson', density=True)
            axes[4].set_title("Error Distribution (in Mask)")
            axes[4].set_xlabel("Absolute Error (K)")
            axes[4].set_ylabel("Density")
            axes[4].grid(True, alpha=0.5)
            # --------------------------------------------------------------------

            fig.colorbar(im, ax=axes[0:3], orientation='horizontal', label='LST (K)', shrink=0.8, pad=0.08)
            fig.colorbar(im_diff, ax=axes[3], orientation='horizontal', label='Difference (K)', shrink=0.8, pad=0.08)

            writer.add_figure(f'Validation Preview/Example_{i+1}', fig, global_step=epoch)

            plot_path = plot_dir / f"epoch_{epoch+1}_preview_{i+1}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

def check_tensor(name, tensor):
    """Helper function to check for NaN or Inf in a tensor."""
    if tensor is None: return False
    is_nan = torch.isnan(tensor).any()
    is_inf = torch.isinf(tensor).any()
    if is_nan or is_inf:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FATAL: Problem detected in tensor: '{name}'")
        if is_nan: print("  - Contains NaN values.")
        if is_inf: print("  - Contains Inf values.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return True
    return False

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        self.features = nn.ModuleList(layers)
        self.final_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        feature_maps = []
        for layer in self.features:
            x = layer(x)
            feature_maps.append(x)
        final_output = self.final_conv(x)
        return final_output, feature_maps

class NLayerDiscriminator(nn.Module):
    """
    Defines the NLayerDiscriminator architecture from the LaMa paper (Listing 2).
    This is a specific PatchGAN with a fixed structure.
    """
    def __init__(self, input_nc=1, ndf=64):
        super().__init__()

        # All layers use a 4x4 kernel and LeakyReLU
        # Note the consistent use of padding=2 as specified in the paper

        # model0: Input layer (Conv -> LeakyReLU)
        self.model0 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # model1: 64 -> 128 (Conv -> BatchNorm -> LeakyReLU)
        self.model1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # model2: 128 -> 256 (Conv -> BatchNorm -> LeakyReLU)
        self.model2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # model3: 256 -> 512 (Conv -> BatchNorm -> LeakyReLU)
        self.model3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=2),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # model4: 512 -> 512 with stride 1 (Conv -> BatchNorm -> LeakyReLU)
        self.model4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # model5: Output layer, 512 -> 1 with stride 1 (Conv)
        self.model5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=2)
        )

    def forward(self, x):
        feature_maps = []

        x = self.model0(x)
        feature_maps.append(x)

        x = self.model1(x)
        feature_maps.append(x)

        x = self.model2(x)
        feature_maps.append(x)

        x = self.model3(x)
        feature_maps.append(x)

        x = self.model4(x)
        feature_maps.append(x)

        final_output = self.model5(x)
        # We don't append the final output to feature_maps

        return final_output, feature_maps

class PhysicalMSELoss(nn.Module):
    def __init__(self, min_k, max_k):
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k

    def forward(self, y_pred, y_true, mask):
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # --- THE FIX: Force Float32 to prevent AMP overflow ---
        pred_f32 = y_pred.float()
        true_f32 = y_true.float()

        # 1. Un-normalize back to physical Kelvin
        pred_kelvin = pred_f32 * (self.max_k - self.min_k) + self.min_k
        true_kelvin = true_f32 * (self.max_k - self.min_k) + self.min_k

        # 2. Extract ONLY the pixels inside the missing mask hole
        mask_bool = mask.bool()
        pred_inpainted = pred_kelvin[mask_bool]
        true_inpainted = true_kelvin[mask_bool]

        # 3. Calculate actual physical MSE
        loss = F.mse_loss(pred_inpainted, true_inpainted)

        return loss
"""
Perceptual loss with VGG19
"""

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Load the pre-trained VGG model
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.feature_layers = nn.Sequential(*list(vgg.children())[:35])

        # 2. Define the "Magic Numbers" that VGG expects (ImageNet Stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, predicted_img, ground_truth_img):
        # --- INPUT IS 1 CHANNEL (e.g., [16, 1, 256, 256]) ---

        # 3. STEP A: Fix the Shape
        # Repeat the 1 channel 3 times to mimic an RGB image
        # New shape: [16, 3, 256, 256]
        pred_3ch = predicted_img.repeat(1, 3, 1, 1)
        gt_3ch = ground_truth_img.repeat(1, 3, 1, 1)

        # 4. STEP B: Fix the Values (Normalization)
        # This shifts your [0, 1] data to the range VGG learned on (approx [-2, 2])
        pred_norm = (pred_3ch - self.mean) / self.std
        gt_norm = (gt_3ch - self.mean) / self.std

        # 5. Compute Loss
        predicted_features = self.feature_layers(pred_norm)
        ground_truth_features = self.feature_layers(gt_norm)
        return F.l1_loss(predicted_features, ground_truth_features)


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()
    def forward(self, real_features, fake_features):
        loss = 0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += self.criterion(fake_feat, real_feat)
        return loss

def compute_gradient_loss(y_pred, y_true, ndvi, mask):
    """
    Computes the L1 loss on the gradients of the images.
    This loss encourages the predicted image to have a similar texture and structure
    to the ground truth and the provided NDVI channel.
    """
    # Define Sobel filters for X and Y gradients
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=y_pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=y_pred.device).view(1, 1, 3, 3)

    # --- Compute Gradients ---
    # We apply padding to keep the image size the same
    def get_gradients(img):
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    # Gradients for the predicted and ground truth LST
    grad_pred = get_gradients(y_pred)
    grad_true = get_gradients(y_true)

    # Gradient for the NDVI channel
    grad_ndvi = get_gradients(ndvi)

    # --- Calculate Loss only on the masked region ---
    # Loss 1: Ensure the inpainted texture matches the real texture
    loss_texture = F.l1_loss(grad_pred * mask, grad_true * mask)

    # Loss 2: Ensure the inpainted texture is guided by the NDVI structure
    loss_structure = F.l1_loss(grad_pred * mask, grad_ndvi * mask)

    # Combine the losses
    return loss_texture + loss_structure #


class LSTGradientLoss(nn.Module):
    """
    Computes the L1 loss on the spatial gradients of the LST images.
    This encourages the predicted image to have a similar texture to the ground truth.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        # Define Sobel filters ONCE during initialization for efficiency
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = sobel_x.to(self.device)
        self.sobel_y = sobel_y.to(self.device)

    def get_gradients(self, img):
        """Calculates the magnitude of the spatial gradients for an image batch."""
        # Apply padding to keep the image size the same after convolution
        grad_x = F.conv2d(img, self.sobel_x, padding=1)
        grad_y = F.conv2d(img, self.sobel_y, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)

    def forward(self, y_pred, y_true, mask):
        """
        Calculates the gradient loss.

        Args:
            y_pred (torch.Tensor): The generated output from the model.
            y_true (torch.Tensor): The ground truth image.
            mask (torch.Tensor): The mask of the inpainted region.
        """
        grad_pred = self.get_gradients(y_pred)
        grad_true = self.get_gradients(y_true)

        # Calculate L1 loss, but only on the gradients within the masked (inpainted) region.
        loss = F.l1_loss(grad_pred * mask, grad_true * mask)

        return loss


# -----------------------------------------------------------------------------
# TRAINER CLASS
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Directories and Logging
        self.output_dir = Path(config['output_dir'])
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.num_workers = config.get('num_workers', 4)
        self.amp_device = self.device.type  # 'cuda' or 'cpu' for torch.autocast / GradScaler

        # DataLoaders
        self._init_dataloaders()
        self._init_preview_dataloader()

        self.writer = SummaryWriter(f"runs/{config['run_name']}")

        # Models
        # The input data has 17 data channels + 1 mask channel = 18 total
        active_count = len(self.config.get('active_data_indices', list(range(12))))
        # config['generator']: 'ffc_lama' (default) or 'swin_unet'; see src/models.py
        print(f"Generator: {self.config.get('generator', 'ffc_lama')}")
        self.generator = build_generator(self.config, in_channels=active_count + 1, out_channels=1).to(self.device)
        # self.discriminator = PatchGANDiscriminator(
        #     n_layers=config['discriminator']['n_layers'],
        #     ndf=config['discriminator']['ndf']
        # ).to(self.device)
        print("Initializing NLayerDiscriminator from the LaMa paper.")
        self.discriminator = NLayerDiscriminator(input_nc=1, ndf=64).to(self.device)
        print("\n" + "="*60)
        print("--- Generator Architecture ---")
        print(self.generator)
        save_model_architecture(self.generator, self.output_dir / "generator_architecture.txt")
        print("\n" + "="*60)
        print("--- Discriminator Architecture ---")
        print(self.discriminator)
        save_model_architecture(self.discriminator, self.output_dir / "discriminator_architecture.txt")
        print("="*60 + "\n")
        self.gradient_loss_fn = LSTGradientLoss(device=self.device)
        self.perceptual_loss_fn = PerceptualLoss().to(self.device)
        self.feature_matching_loss_fn = FeatureMatchingLoss()
        self.physical_mse_fn = PhysicalMSELoss(
            min_k=config['normalization']['lst']['min'],
            max_k=config['normalization']['lst']['max']
        ).to(self.device)

        # Optimizers
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config['learning_rate_g'], betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config['learning_rate_d'], betas=(0.5, 0.999))

        self.scheduler_type = config.get('scheduler_type', 'plateau')

        if self.scheduler_type == "plateau":
            # NOTE: `verbose=` was removed from ReduceLROnPlateau in recent PyTorch versions (TypeError);
            # the learning rate is logged to TensorBoard instead.
            self.scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_g, mode='min', factor=0.5, patience=5, min_lr=1e-6)
            self.scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_d, mode='min', factor=0.5, patience=5, min_lr=1e-7)

        elif self.scheduler_type == "cosine_warmup":
            warmup_epochs = config.get('warmup_epochs', 5)
            total_epochs = config['num_epochs']
            print(f"Scheduler: Warmup + Cosine Annealing (Warmup: {warmup_epochs} epochs)")

            warmup_g = optim.lr_scheduler.LinearLR(self.optimizer_g, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            warmup_d = optim.lr_scheduler.LinearLR(self.optimizer_d, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)

            cosine_g = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_g, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)
            cosine_d = optim.lr_scheduler.CosineAnnealingLR(self.optimizer_d, T_max=(total_epochs - warmup_epochs), eta_min=1e-7)

            self.scheduler_g = optim.lr_scheduler.SequentialLR(self.optimizer_g, schedulers=[warmup_g, cosine_g], milestones=[warmup_epochs])
            self.scheduler_d = optim.lr_scheduler.SequentialLR(self.optimizer_d, schedulers=[warmup_d, cosine_d], milestones=[warmup_epochs])

        else:
            raise ValueError(f"Unknown scheduler_type in config: {self.scheduler_type}")
        # AMP: torch.amp.GradScaler exists from PyTorch 2.3; fall back to torch.cuda.amp on older versions (e.g. 2.1)
        if hasattr(torch.amp, "GradScaler"):
            self.scaler_g = torch.amp.GradScaler(self.amp_device, enabled=config['use_amp'])
            self.scaler_d = torch.amp.GradScaler(self.amp_device, enabled=config['use_amp'])
        else:
            self.scaler_g = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
            self.scaler_d = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        # squared=True computes MSE, squared=False computes RMSE
        self.mse_metric = MeanSquaredError(squared=True).to(self.device)
        self.rmse_metric = MeanSquaredError(squared=False).to(self.device)

        #MAE
        self.mae_metric = MeanAbsoluteError().to(self.device)

        # For SSIM, data_range is important since our data is normalized to [0, 1]
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

        # LPIPS requires a pre-trained network. 'vgg' is common for higher accuracy.
        self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)

        # FID compares distributions and requires a feature extractor.
        # The 'feature' param selects an InceptionV3 layer. 64 is faster, 2048 is standard.
        self.fid_metric = FrechetInceptionDistance(feature=64).to(self.device)

        # State
        self.start_epoch = 0
        self.best_val_rmse = float('inf')


    def _init_preview_dataloader(self):
        """Initializes a separate dataloader for generating specific preview plots."""
        if self.config.get('preview_tiles'):
            print("Initializing dedicated preview dataloader.")
            # Preview tiles are expected in the validation set directory
            val_tile_dir = Path(self.config['split_data_dir']) / "val" / "tiles"
            requested = [val_tile_dir / fname for fname in self.config['preview_tiles']]
            preview_tile_paths = [p for p in requested if p.exists()]
            missing = [p.name for p in requested if not p.exists()]
            if missing:
                print(f"WARNING: preview tiles not found in {val_tile_dir} and ignored: {missing}")
            if not preview_tile_paths:
                print("No preview tiles available; using standard validation loader for plots.")
                self.preview_loader = None
                return

            preview_dataset = LSTInpaintingDataset(
                training_tile_dir=val_tile_dir,
                mask_tile_dir=self.config['mask_dir'],
                active_indices=self.config.get('active_data_indices'),  # must match the model input channels
                tile_list=preview_tile_paths
            )

            self.preview_loader = DataLoader(
                preview_dataset,
                batch_size=len(preview_tile_paths),  # all preview tiles in one batch
                shuffle=False,
                num_workers=self.num_workers
            )
        else:
            print("No specific preview tiles listed; using standard validation loader for plots.")
            self.preview_loader = None

    def _init_dataloaders(self):
        train_tile_dir = Path(self.config['split_data_dir']) / "train" / "tiles"
        val_tile_dir = Path(self.config['split_data_dir']) / "val" / "tiles"

        train_dataset = LSTInpaintingDataset(training_tile_dir=train_tile_dir,
                                             mask_tile_dir=self.config['mask_dir'],
                                             active_indices=self.config.get('active_data_indices'))

        self.train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=self.num_workers, pin_memory=torch.cuda.is_available())

        val_dataset = LSTInpaintingDataset(training_tile_dir=val_tile_dir,
                                           mask_tile_dir=self.config['mask_dir'],
                                           active_indices=self.config.get('active_data_indices'))
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False,
                                     num_workers=self.num_workers, pin_memory=torch.cuda.is_available())
        print("Dataloaders initialized.")

    def _load_checkpoint(self):
        if self.config.get('resume_checkpoint'):
            ckpt_path = Path(self.config['resume_checkpoint'])
            if ckpt_path.exists():
                print(f"Resuming training from checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator_state_dict'])
                self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
                self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

                if 'scheduler_g_state_dict' in checkpoint:
                    self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
                    self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])

                self.start_epoch = checkpoint.get('epoch', 0) + 1
                # self.best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
                self.best_val_rmse = checkpoint.get('best_val_rmse', float('inf'))
                print(f"Resumed from epoch {self.start_epoch}.")
            else:
                print(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")


    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'best_val_rmse': self.best_val_rmse,
            'config': self.config
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, path)
            print(f"Saved new best model to {path}")
        if (epoch + 1) % self.config['save_checkpoint_every_n_epochs'] == 0:
            path = self.checkpoint_dir / f"epoch_{epoch+1}.pth"
            torch.save(state, path)
            print(f"Saved checkpoint to {path}")

    def train(self):
        self._load_checkpoint()
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            self._train_one_epoch(epoch)
            val_rmse = self._validate_one_epoch(epoch)

            if self.scheduler_type == "plateau":
                self.scheduler_g.step(val_rmse)
                self.scheduler_d.step(val_rmse)

            elif self.scheduler_type == "cosine_warmup":
                # Cosine just needs to be told an epoch has finished
                self.scheduler_g.step()
                self.scheduler_d.step()

            # Log the LR so you can see the drop in TensorBoard
            current_lr = self.optimizer_g.param_groups[0]['lr']
            self.writer.add_scalar('LR/Generator', current_lr, epoch)

            is_best = val_rmse < self.best_val_rmse  # lower is better
            if is_best:
                self.best_val_rmse = val_rmse

            self._save_checkpoint(epoch, is_best)


            # This try/except block prevents plotting errors from crashing the training
            print("Generating validation preview plots...")
            try:

                plot_loader = self.preview_loader if self.preview_loader else self.val_loader

                generate_validation_plots(
                    self.writer,
                    self.generator,
                    plot_loader,  # was self.val_loader, so preview_tiles were never used
                    self.device,
                    epoch,
                    self.output_dir,
                    self.config,
                    num_plots=min(4, len(plot_loader.dataset))
                )
            except Exception as e:
                print(f"WARNING: Could not generate validation plots for epoch {epoch+1}. Error: {e}")

        self.writer.close()
        print("Training complete.")

    def _train_one_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()

        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{self.config['num_epochs']} [Training]")

        for batch_idx, batch in progress_bar:
            model_input = batch['image'].to(self.device)
            ground_truth = batch['gt_image'].to(self.device)
            ground_truth.requires_grad = True
            mask = model_input[:, -1:, :, :]

            # --- Train Discriminator (in full float32 precision) ---
            if batch_idx % 2 == 0:
                self.optimizer_d.zero_grad(set_to_none=True)

                with torch.autocast(device_type=self.amp_device, enabled=self.config['use_amp']):
                    generated_output = self.generator(model_input)
                    composite_output = ground_truth * (1 - mask) + generated_output * mask

                    real_output, _ = self.discriminator(ground_truth)
                    fake_output, _ = self.discriminator(composite_output.detach())

                    loss_d_real = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))
                    loss_d_fake = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))

                    grad_real, = torch.autograd.grad(outputs=real_output.sum(), inputs=ground_truth, create_graph=True)
                    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

                    loss_d = loss_d_real + loss_d_fake + (self.config['gamma'] * grad_penalty)

                # loss_d.backward()
                # # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                # self.optimizer_d.step()
                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()

            # --- Train Generator (in full float32 precision) ---
            self.optimizer_g.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.amp_device, enabled=self.config['use_amp']):
                generated_output = self.generator(model_input)
                composite_output = ground_truth * (1 - mask) + generated_output * mask
                fake_output_for_g, fake_features = self.discriminator(composite_output)
                _, real_features = self.discriminator(ground_truth.detach())

                loss_g_adv = F.binary_cross_entropy_with_logits(fake_output_for_g, torch.ones_like(fake_output_for_g))
                loss_g_feat = self.feature_matching_loss_fn(real_features, fake_features)
                loss_g_perc = self.perceptual_loss_fn(composite_output, ground_truth)


                recon_type = self.config.get('reconstruction_loss_type', 'hybrid_q4')

                if recon_type == "physical_mse":
                    loss_g_recon = self.physical_mse_fn(generated_output, ground_truth, mask)

                #--- L1 Reconstruction Loss ---
                elif recon_type == "standard_mae":
                    #UNweighted L1 loss

                    #This calculates the direct pixel-wise error ONLY on the inpainted region
                    loss_g_recon = F.l1_loss(generated_output * mask, ground_truth * mask)

                    # ndvi_channel = model_input[:, 16:17, :, :]
                    # loss_g_grad = compute_gradient_loss(generated_output, ground_truth, ndvi_channel, mask)
                    # loss_g_grad = self.gradient_loss_fn(generated_output, ground_truth, mask)
                elif recon_type == "hybrid_q4":
                    # --- HYBRID Q4-TARGETED RECONSTRUCTION LOSS ---
                    # 1. Calculate the raw difference (Error Map)
                    #diff = (generated_output - ground_truth) * mask
                    diff_l1 = torch.abs(generated_output - ground_truth) * mask

                    # 2. Identify the top 25% of pixels (Q4) in the ground truth
                    # We do this per-batch to ensure the model always focuses on the hottest areas
                    with torch.no_grad():
                        q75_threshold = torch.quantile(ground_truth[mask.bool()], 0.75)
                        is_q4 = (ground_truth > q75_threshold) & mask.bool()
                        is_rest = (~is_q4) & mask.bool()

                    # 3. Calculate separate L1 means
                    loss_q4 = diff_l1[is_q4].mean() if is_q4.any() else 0.0
                    loss_others = diff_l1[is_rest].mean() if is_rest.any() else 0.0

                    # 4. Combine with a small boost for Q4
                    # This focuses the model on the 342K peaks without breaking the 'safe' scenes
                    loss_g_recon = loss_others + (2.0 * loss_q4)
                else:
                    raise ValueError(f"Unknown reconstruction loss type in config: {recon_type}")

                loss_g_grad = self.gradient_loss_fn(generated_output, ground_truth, mask)

                loss_g = (self.config['kappa'] * loss_g_adv) + \
                       (self.config['beta'] * loss_g_feat) + \
                       (self.config['alpha'] * loss_g_perc) + \
                       (self.config['delta'] * loss_g_grad)  + \
                       (self.config['lambda'] * loss_g_recon)



            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()

            # loss_g.backward()
            # # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            # self.optimizer_g.step()

            progress_bar.set_postfix(Loss_D=f"{loss_d.item():.4f}", Loss_G=f"{loss_g.item():.4f}")
            if self.writer:
                self.writer.add_scalar('Loss_Batch/Generator', loss_g.item(), epoch * len(self.train_loader) + batch_idx)
                self.writer.add_scalar('Loss_Batch/Discriminator', loss_d.item(), epoch * len(self.train_loader) + batch_idx)


    def _validate_one_epoch(self, epoch):
        self.generator.eval()
        self.discriminator.eval()

        val_g_loss = 0.0
        total_lpips = 0.0 # Accumulator for LPIPS score

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Validation]")):
                model_input = batch['image'].to(self.device)
                ground_truth = batch['gt_image'].to(self.device)
                mask = model_input[:, -1:, :, :]

                with torch.autocast(device_type=self.amp_device, enabled=self.config['use_amp']):
                    generated_output = self.generator(model_input)
                    composite_output = ground_truth * (1 - mask) + generated_output * mask

                    # --- Calculate metrics on the inpainted region only ---
                    mask_bool = mask.bool()
                    inpainted_pixels = generated_output[mask_bool]
                    original_pixels = ground_truth[mask_bool]
                    self.mse_metric.update(inpainted_pixels, original_pixels)
                    self.rmse_metric.update(inpainted_pixels, original_pixels)

                    # --- 2. Full Image Metrics (MAE/SSIM) ---
                    self.mae_metric.update(composite_output, ground_truth)
                    self.ssim_metric.update(composite_output, ground_truth)

                    # --- 3. Perceptual Metric (LPIPS) ---
                    # LPIPS expects images in range [-1, 1] and 3 channels
                    composite_lpips = (composite_output * 2 - 1).repeat(1, 3, 1, 1)
                    gt_lpips = (ground_truth * 2 - 1).repeat(1, 3, 1, 1)
                    total_lpips += self.lpips_metric(composite_lpips, gt_lpips).sum().item()

                    # --- 4. Distribution Metric (FID) ---
                    # FID expects uint8 images in range [0, 255] and 3 channels
                    gt_uint8 = (ground_truth.repeat(1, 3, 1, 1).clamp(0, 1) * 255).to(torch.uint8)
                    composite_uint8 = (composite_output.repeat(1, 3, 1, 1).clamp(0, 1) * 255).to(torch.uint8)
                    self.fid_metric.update(gt_uint8, real=True)
                    self.fid_metric.update(composite_uint8, real=False)



                    # --- 5. Generator Validation Loss ---
                    fake_output_for_g, fake_features = self.discriminator(composite_output)
                    _, real_features = self.discriminator(ground_truth)
                    loss_g_adv = F.binary_cross_entropy_with_logits(fake_output_for_g, torch.ones_like(fake_output_for_g))
                    loss_g_feat = self.feature_matching_loss_fn(real_features, fake_features)
                    loss_g_perc = self.perceptual_loss_fn(composite_output, ground_truth)


                    recon_type = self.config.get('reconstruction_loss_type', 'hybrid_q4')
                    # # Add the L1 loss to the validation monitoring ---
                    if recon_type == "standard_mae":
                        loss_g_recon = F.l1_loss(generated_output * mask, ground_truth * mask)

                    elif recon_type == "physical_mse":
                        loss_g_recon = self.physical_mse_fn(generated_output, ground_truth, mask)

                    elif recon_type == "hybrid_q4":
                        # # Hybrid Reconstruction Logic
                        # diff = (generated_output - ground_truth) * mask
                        # q75_threshold = torch.quantile(ground_truth[mask.bool()], 0.75)
                        # is_q4 = (ground_truth > q75_threshold) & mask.bool()
                        # is_rest = (~is_q4) & mask.bool()

                        # loss_q4 = (diff[is_q4]**2).mean() if is_q4.any() else 0.0
                        # loss_others = torch.abs(diff[is_rest]).mean() if is_rest.any() else 0.0
                        # loss_g_l1 = loss_others + (2.0 * loss_q4)

                        # Hybrid Reconstruction Logic (Corrected L1)
                        diff_l1 = torch.abs(generated_output - ground_truth) * mask

                        q75_threshold = torch.quantile(ground_truth[mask.bool()], 0.75)
                        is_q4 = (ground_truth > q75_threshold) & mask.bool()
                        is_rest = (~is_q4) & mask.bool()

                        loss_q4 = diff_l1[is_q4].mean() if is_q4.any() else 0.0
                        loss_others = diff_l1[is_rest].mean() if is_rest.any() else 0.0

                        loss_g_recon = loss_others + (2.0 * loss_q4)
                    else:
                        raise ValueError(f"Unknown reconstruction loss type in config: {recon_type}")

                    # --- Calculate the Gradient Loss during validation ---
                    loss_g_grad = self.gradient_loss_fn(generated_output, ground_truth, mask)

                    # --- Add the gradient loss to the total validation loss ---
                    loss_g = (self.config['kappa'] * loss_g_adv) + \
                           (self.config['beta'] * loss_g_feat) + \
                           (self.config['alpha'] * loss_g_perc) + \
                           (self.config['delta'] * loss_g_grad)  + \
                           (self.config['lambda'] * loss_g_recon)




                    val_g_loss += loss_g.item()

                # -------------------------------------------------------------

                # --- PLOT BATCH LOSS ---
                if self.writer:
                    global_step = epoch * len(self.val_loader) + i
                    self.writer.add_scalar('Loss_Batch_Val/Generator', loss_g.item(), global_step)

                # if i == 0 and self.writer:
                #     # Your image logging code to TensorBoard goes here
                #     pass

        # Compute and log average values
        avg_val_g_loss = val_g_loss / len(self.val_loader)
        avg_lpips = total_lpips / len(self.val_loader.dataset)
        final_mse = self.mse_metric.compute()
        final_rmse = self.rmse_metric.compute()
        final_mae = self.mae_metric.compute()
        final_ssim = self.ssim_metric.compute()
        final_fid = self.fid_metric.compute()

        if self.writer:
            self.writer.add_scalar('Loss_Epoch/Validation_Generator', avg_val_g_loss, epoch)
            self.writer.add_scalar('Metrics/MSE_inpainted', final_mse, epoch)
            self.writer.add_scalar('Metrics/RMSE_inpainted', final_rmse, epoch)
            self.writer.add_scalar('Metrics/MAE_full', final_mae, epoch)
            self.writer.add_scalar('Metrics/SSIM_full', final_ssim, epoch)
            self.writer.add_scalar('Metrics/LPIPS_full', avg_lpips, epoch)
            self.writer.add_scalar('Metrics/FID_full', final_fid, epoch)

        print(f"Epoch {epoch+1} Results:")
        print(f"  - Val Loss: {avg_val_g_loss:.4f} | RMSE (inpainted): {final_rmse.item():.4f}")
        print(f"  - MAE: {final_mae.item():.4f} | SSIM: {final_ssim.item():.4f} | LPIPS: {avg_lpips:.4f} | FID: {final_fid.item():.2f}")

        self.mse_metric.reset()
        self.rmse_metric.reset()
        self.mae_metric.reset()
        self.ssim_metric.reset()
        self.fid_metric.reset()

        return final_rmse.item()

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LaMa model for LST inpainting.")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config)
    trainer.train()