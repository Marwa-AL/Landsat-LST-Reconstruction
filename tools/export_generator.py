"""
Export a training checkpoint to a small, generator-only file for publishing
(e.g. as a GitHub Release asset). Discriminator and optimizer states are dropped,
which shrinks the file roughly 3-4x; evaluate.py / inference.py load it unchanged.

Usage (from the repository root):
    python tools/export_generator.py \
        --checkpoint outputs/full_multimodal/checkpoints/best_model.pth \
        --generator ffc_lama \
        --out checkpoints/ffc_gan_generator.pth

    --generator: ffc_lama (Multimodal FFC-GAN) or swin_unet (SwinUNet-GAN)
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models import build_generator  # noqa: E402

KEEP_CONFIG_KEYS = ("generator", "active_data_indices", "normalization", "min_kelvin", "max_kelvin",
                    "reconstruction_loss_type", "kappa", "alpha", "beta", "gamma", "lambda", "delta",
                    "batch_size", "num_epochs", "learning_rate_g", "learning_rate_d", "scheduler_type")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Training checkpoint (.pth) with 'generator_state_dict'")
    ap.add_argument("--generator", required=True, choices=["ffc_lama", "swin_unet"])
    ap.add_argument("--out", required=True, help="Output file, e.g. checkpoints/ffc_gan_generator.pth")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # own, trusted file
    state = ckpt["generator_state_dict"]
    train_cfg = dict(ckpt.get("config", {}))
    train_cfg["generator"] = args.generator
    active = train_cfg.get("active_data_indices", list(range(20)))

    # Verify that the weights load into the architecture before writing anything
    model = build_generator(train_cfg, in_channels=len(active) + 1, out_channels=1)
    model.load_state_dict(state)

    best = ckpt.get("best_val_rmse")
    out = {
        "generator_state_dict": state,
        "config": {k: train_cfg[k] for k in KEEP_CONFIG_KEYS if k in train_cfg},
        "epoch": ckpt.get("epoch"),
        "best_val_rmse": float(best) if best is not None else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n = sum(v.numel() for v in state.values())
    print(f"Saved {args.out}: {args.generator}, {n / 1e6:.1f} M parameters, "
          f"{len(active)} input channels (+ mask), epoch {out['epoch']}, "
          f"size {Path(args.out).stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
