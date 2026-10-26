#!/usr/bin/env bash
# End-to-end smoke test on a tiny synthetic dataset (no real data needed).
# Run from the repository root:   bash tests/smoke_test.sh            (Multimodal FFC-GAN)
#                                  bash tests/smoke_test.sh swin_unet  (SwinUNet-GAN)
# Takes roughly 20-30 minutes on a laptop CPU (much less on a GPU).
# Needs internet on the first run to download VGG19 / LPIPS / Inception weights.
set -euo pipefail

GENERATOR=${1:-ffc_lama}
WORK=smoke_run_${GENERATOR}
rm -rf "$WORK" && mkdir -p "$WORK"
cp -r src configs "$WORK"/
[ -d third_party ] && cp -r third_party "$WORK"/
cd "$WORK"

python ../tests/make_synthetic_data.py --out data

python src/generate_hillshades.py       --config configs/data_config.yaml
python src/mask_tile_generator.py       --config configs/data_config.yaml
python src/generating_training_tiles.py --config configs/data_config.yaml
python src/split_dataset.py             --config configs/data_config.yaml

# Small, fast training config derived from the real one
GENERATOR=$GENERATOR python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/config.yaml"))
import os
c.update(generator=os.environ["GENERATOR"], batch_size=4, num_epochs=1, num_workers=0, preview_tiles=[],
         checkpoint_path=c["output_dir"] + "/checkpoints/best_model.pth")
yaml.safe_dump(c, open("configs/smoke_config.yaml", "w"), sort_keys=False)
PY

python src/train.py     --config configs/smoke_config.yaml
python src/evaluate.py  --config configs/smoke_config.yaml
python src/inference.py --config configs/smoke_config.yaml

echo "SMOKE TEST PASSED - outputs in $WORK/outputs"
