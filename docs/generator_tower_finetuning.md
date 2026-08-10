# Generator-tower image-editing fine-tuning

Cosmos3 generator fine-tuning supports all three public omni checkpoints:

| Checkpoint | Generator update | Recipe |
| --- | --- | --- |
| Cosmos3-Edge | Full generator pathway | `image_edit_sft_edge.toml` |
| Cosmos3-Nano | Full generator pathway | `image_edit_sft_nano.toml` |
| Cosmos3-Super | LoRA on `q/k/v/o_proj_moe_gen` | `image_edit_sft_super.toml` |

The recipes use paired image editing: each JSONL record supplies a conditioned
source image, a generated target image, and an edit instruction. Source and
target paths may be absolute or relative to the manifest.

```json
{"id":"pair-0001","source":"images/clean.png","target":"images/edited.png","instruction":"Add a scratch."}
```

Both dimensions must be multiples of 16. The provided recipes use 848×480 and
one pair per microbatch. Validation evaluates the same flow-matching objective
with a fixed noise sequence, so losses are comparable between checkpoints.

## UC3 / AnomalyGen adapter

The AnomalyGen UC3 archive is not paired. The adapter splits clean identities
and real defect/mask exemplars before creating deterministic cross-products.
It composites only masked real-defect pixels onto the clean source and writes
192 training pairs, 12 held-out validation pairs, and 15 fixed comparison
cases for the supplied 20-clean/15-defect archive.

```bash
python tools/prepare_uc3_image_editing.py \
  --archive /path/to/UC3_data.zip \
  --output-dir /data/uc3-paired \
  --width 848 --height 480 --feather-radius 2
```

The output retains the original AnomalyGen layout under
`raw/UC3_data/`, allowing the same data root to be used by the AnomalyGen
baseline and evaluator. `dataset_summary.json` records archive and manifest
SHA-256 hashes.

## Checkpoint preparation and training

Convert the selected public checkpoint to DCP as described in
[training.md](training.md#step-2--prepare-checkpoint):

```bash
python -m cosmos_framework.scripts.convert_model_to_dcp \
  --checkpoint-path Cosmos3-Edge \
  -o /checkpoints/Cosmos3-Edge
```

Then launch the paired recipe:

```bash
export DATASET_PATH=/data/uc3-paired
export BASE_CHECKPOINT_PATH=/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=/checkpoints/wan22_vae/Wan2.2_VAE.pth
export IMAGINAIRE_OUTPUT_ROOT=/results/train
export TAO_STATUS_FILE=/results/status.json

NPROC_PER_NODE=8 bash examples/launch_sft_image_edit.sh edge
```

Use `nano` or `super` as the final argument for the other checkpoints. Hydra
overrides after the model name take precedence over TOML values:

```bash
bash examples/launch_sft_image_edit.sh edge \
  trainer.max_iter=5 \
  trainer.max_val_iter=1 \
  trainer.validation_iter=2 \
  checkpoint.save_iter=5
```

For multi-node launches, set `NNODES`, `NODE_RANK`, and `MASTER_ADDR`; keep
`checkpoint.dcp_async_mode_enabled=false` so a completed status never races an
unfinished distributed checkpoint.

## Validation

Use native `image2image` inference on the fixed `comparison.jsonl` cases. For
AnomalyGen-compatible scoring, normalize generated images under
`reconstructed_image/` and their masks under `original_mask/` with names such
as `Phone+scratch_0000.png`.

Report:

- training and deterministic validation loss curves;
- AnomalyGen DINOv2 `nn_score` as the primary semantic-defect KPI;
- `mnn_score` and FID as diagnostics (FID is unreliable with fewer than ten
  real images per type);
- outside-mask identity MAE/PSNR;
- source / Cosmos3 / AnomalyGen / real-reference contact sheets.

The UC3 composite target is a practical adapter, not an exact reproduction of
AnomalyGen's mask-conditioned inpainting architecture. Pixel accuracy against
the original anomaly image is therefore not a meaningful primary metric.
