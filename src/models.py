"""
Generator factory: builds the network selected by `generator` in the config.

    generator: "ffc_lama"   -> Multimodal FFC-GAN generator: LaMa with Fast Fourier Convolutions (src/lama_model.py)
    generator: "swin_unet"  -> SwinUNet-GAN generator: Swin-UNet network (Cao et al., 2022), code from
                               the third_party/Swin-Unet git submodule

Both are trained with the same GAN setup (NLayer PatchGAN discriminator, same
losses) in src/train.py; only the generator differs.
"""
import sys
from pathlib import Path

from lama_model import LaMaGenerator

SWIN_UNET_DIR = Path(__file__).resolve().parents[1] / "third_party" / "Swin-Unet"

# Swin-UNet hyper-parameters used for the paper's SwinUNet-GAN (Swin-T style encoder, 256x256 tiles)
SWIN_UNET_SETTINGS = dict(
    img_size=256, patch_size=4, embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
    window_size=8, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, ape=False, patch_norm=True,
    drop_rate=0.0, drop_path_rate=0.1,
)


def _build_swin_unet(in_channels, out_channels):
    if not (SWIN_UNET_DIR / "networks" / "vision_transformer.py").exists():
        raise ImportError(
            f"Swin-UNet code not found in {SWIN_UNET_DIR}.\n"
            "Fetch it with:  git submodule update --init --recursive")
    if str(SWIN_UNET_DIR) not in sys.path:
        sys.path.insert(0, str(SWIN_UNET_DIR))
    import ml_collections
    from networks.vision_transformer import SwinUnet

    s = SWIN_UNET_SETTINGS
    cfg = ml_collections.ConfigDict()
    cfg.DATA = ml_collections.ConfigDict()
    cfg.DATA.IMG_SIZE = s["img_size"]
    cfg.MODEL = ml_collections.ConfigDict()
    cfg.MODEL.TYPE = "swin"
    cfg.MODEL.NUM_CLASSES = out_channels
    cfg.MODEL.DROP_RATE = s["drop_rate"]
    cfg.MODEL.DROP_PATH_RATE = s["drop_path_rate"]
    cfg.MODEL.SWIN = ml_collections.ConfigDict()
    cfg.MODEL.SWIN.PATCH_SIZE = s["patch_size"]
    cfg.MODEL.SWIN.IN_CHANS = in_channels
    cfg.MODEL.SWIN.EMBED_DIM = s["embed_dim"]
    cfg.MODEL.SWIN.DEPTHS = s["depths"]
    cfg.MODEL.SWIN.NUM_HEADS = s["num_heads"]
    cfg.MODEL.SWIN.WINDOW_SIZE = s["window_size"]
    cfg.MODEL.SWIN.MLP_RATIO = s["mlp_ratio"]
    cfg.MODEL.SWIN.QKV_BIAS = s["qkv_bias"]
    cfg.MODEL.SWIN.QK_SCALE = s["qk_scale"]
    cfg.MODEL.SWIN.APE = s["ape"]
    cfg.MODEL.SWIN.PATCH_NORM = s["patch_norm"]
    cfg.TRAIN = ml_collections.ConfigDict()
    cfg.TRAIN.USE_CHECKPOINT = False
    return SwinUnet(config=cfg, img_size=s["img_size"], num_classes=out_channels).float()


def build_generator(config, in_channels, out_channels=1):
    """Return the generator named by config['generator'] (default: 'ffc_lama')."""
    name = config.get("generator", "ffc_lama")
    if name == "ffc_lama":
        return LaMaGenerator(input_channels=in_channels, output_channels=out_channels)
    if name == "swin_unet":
        return _build_swin_unet(in_channels, out_channels)
    raise ValueError(f"Unknown generator '{name}' in config (use 'ffc_lama' or 'swin_unet')")
