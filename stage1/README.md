# HAE Stage 1: Autoencoder Training

Training and evaluation code for the Hyperspherical Autoencoder (HAE): a frozen vision foundation model
(DINOv3 ViT-L/16) whose patch embedding is replaced by the Hierarchical Convolutional Patch Embedding (HCPE),
followed by a DC-AE decoder.

Training has two steps:

| Step | Config | Trains | Objective |
| --- | --- | --- | --- |
| Stage 1: alignment | `configs/stage1_alignment.yaml` | HCPE + decoder | L1 + LPIPS + 0.5 · (1 − cos) directional alignment to the frozen teacher |
| Stage 2: GAN | `configs/stage2_gan.yaml` | decoder | L1 + LPIPS + adaptive DinoDisc GAN |

## Layout

```text
stage1/
├── configs/            base.yaml, stage1_alignment.yaml, stage2_gan.yaml, siglip2/
├── hae/                training package (python -m hae.train, python -m hae.evaluate)
│   ├── models/         autoencoder, HCPE, DC-AE decoder, DinoDisc
│   └── siglip2/        optional SigLIP2 encoder
├── scripts/            train.sh, weight preparation, multi-GPU evaluation
├── tests/              CPU tests (pytest)
└── third_party/        licenses of the code this folder derives from
```

## Setup

Run every command from this `stage1/` folder.

```bash
# 1. Environment (install the torch / torchvision build for your CUDA version first)
pip install -r requirements.txt

# 2. DINOv3 source (tested with commit adeee9af6d6559dd2f0d6c51bec03917de5b9619)
git clone https://github.com/facebookresearch/dinov3.git vendor/dinov3

# 3. LPIPS VGG weights (add --fid for the Inception weights used by rFID)
python scripts/prepare_perceptual.py --fid
```

Place the pretrained weights and the data as follows (or change the paths in `configs/base.yaml`):

```text
stage1/
├── weights/
│   ├── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth   # DINOv3 ViT-L/16, from the DINOv3 repository
│   └── dino_deitsmall8_pretrain.pth                    # DINO ViT-S/8 for the discriminator (Stage 2)
└── data/imagenet/
    ├── train/<class>/*.JPEG
    └── val/...                                         # class folders or a flat folder
```

The DINO ViT-S/8 weights are at
`https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth`.
Training never downloads anything by itself.

## Training

```bash
# Stage 1: HCPE + decoder, 25 epochs
GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml

# Stage 2: decoder-only GAN, 10 epochs, starts from the EMA weights of Stage 1
GPUS=0,1,2,3 scripts/train.sh configs/stage2_gan.yaml
```

### Stage 1 schedule

The learning rate and the Adam betas change by phase inside one run (`training.phases`, lengths in epochs).
The Adam moments carry over between phases.

| Epochs | Learning rate | Adam betas |
| --- | --- | --- |
| 0 – 3 | 1e-5 | (0.9, 0.999) |
| 3 – 25 | 1e-4 | (0.9, 0.9) |

### Stage 2 schedule

| Setting | Value |
| --- | --- |
| Generator / discriminator lr | 5.4e-5 / 5.4e-5 |
| Adam betas (both) | (0.5, 0.9) |
| Generator lr warmup | 1 epoch (linear) |
| Discriminator-only warmup | 2 epochs, then the adversarial term turns on |
| Adversarial weight | 0.5 × adaptive weight (last-layer gradient ratio, capped at 10) |
| EMA decay | 0.9975 |
| Length | 10 epochs |

Stage 2 counts in optimizer updates. The values in `configs/stage2_gan.yaml` assume ImageNet-1k at global
batch 512 (1 epoch = 2,502 updates); rescale `lr_warmup_updates`, `gan.warmup_updates`, `max_updates` and
`save_every` if you change the batch size or the dataset.

### Batch size and memory

`global_batch_size` (512) = `micro_batch_size` × GPUs × gradient accumulation. The accumulation is derived
automatically, so any GPU count that divides the batch gives the same training. `micro_batch_size` is 64 in
the configs, which needs very large GPUs (roughly 3.5 GB per image at 256 px in bf16). Lower it with `MICRO`:

```bash
GPUS=0,1,2,3,4,5,6,7 MICRO=16 scripts/train.sh configs/stage1_alignment.yaml
```

In Stage 2 the adaptive adversarial weight is computed per micro batch, so `MICRO` slightly changes its
statistics.

### Resuming and overrides

```bash
# exact resume (same GPU count, data and config)
GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml --resume runs/stage1_alignment/checkpoint_last.pt

# continue a checkpoint with a new lr schedule, Adam betas or GPU count (optimizer, EMA and counters are kept)
GPUS=0,1 scripts/train.sh configs/stage1_alignment.yaml --continue-from runs/stage1_alignment/checkpoint_last.pt \
    --set training.output_dir=runs/stage1_alignment_2gpu

# any config value can be overridden
GPUS=0 scripts/train.sh configs/stage2_gan.yaml --set logging.wandb.mode=disabled --set data.num_workers=4
```

Other options:

- `data.preprocess: adm_center_crop` uses the ADM / RAE center crop (bicubic) instead of the bilinear squash.
- `losses.alignment_type: mse` replaces the directional (cosine) alignment by a feature MSE.
- A phase may end early on a validation plateau:
  `{max_epochs: 22, learning_rate: 1.0e-4, betas: [0.9, 0.9], until_plateau: {patience: 2, min_psnr_gain_db: 0.05, min_lpips_gain: 0.003}}`.

### Outputs

Each run writes to `training.output_dir`:

| File | Content |
| --- | --- |
| `checkpoint_<update>.pt`, `checkpoint_last.pt` | model, EMA, optimizers, discriminator, RNG and data position (one per epoch) |
| `metrics.jsonl` | training losses every `log_every` updates |
| `validation.jsonl` | PSNR / SSIM / LPIPS of the EMA weights on the first 1,000 validation images, every epoch |
| `schedule.json` | phase boundaries and status (Stage 1) |
| `hcpe_warmup.pt`, `hcpe_warmup.json`, `hcpe_warmup.jsonl` | HCPE warmup result and log (only with `training.hcpe_warmup`) |
| `resolved_config.json` | the merged config of the run |

Weights & Biases logging is on by default (`wandb login` first, or `--set logging.wandb.mode=offline|disabled`).

## Evaluation

```bash
python -m hae.evaluate --config configs/stage2_gan.yaml \
    --checkpoint runs/stage2_gan/checkpoint_last.pt --weights ema \
    --data-root data/imagenet/val --device cuda --batch-size 64 --num-workers 8 \
    --lpips --ssim --rfid --real-stats reports/val_real_stats.npz \
    --output-dir reports/stage2_gan
```

This writes `summary.json` (mean PSNR, SSIM, LPIPS-VGG, rFID) and `per_image.json`. Images are resized with
`--preprocess squash_bilinear` (default) or `adm_center_crop`. `scripts/evaluate_sharded.py` splits the same
evaluation over several GPUs; it reuses the real-image statistics that the command above caches in
`--real-stats`.

### Pretrained weights

The released autoencoders are on Hugging Face: [wkdgnsgo/hae](https://huggingface.co/wkdgnsgo/hae)
(`hae_no_smoothing.pt`, `hae_smoothing.pt`). Each file is a complete state dict and is evaluated like a
training checkpoint:

```bash
wget -P weights/hae https://huggingface.co/wkdgnsgo/hae/resolve/main/hae_no_smoothing.pt

python -m hae.evaluate --config configs/stage1_alignment.yaml \
    --checkpoint weights/hae/hae_no_smoothing.pt \
    --data-root data/imagenet/val --device cuda --batch-size 64 --num-workers 8 \
    --lpips --ssim --rfid --output-dir reports/hae_no_smoothing
```

Only the DINOv3 source in `vendor/dinov3` is needed for this; the DINOv3 weights are part of the file.
To start a training stage from a released autoencoder, pass it with `--init-from`.

## HCPE warmup

DINOv3 trains with a randomly initialised HCPE. Encoders whose frozen blocks amplify small changes of the
input embedding, such as SigLIP2 or DINOv2, do not: with a random HCPE the tokens end up far from the
teacher's and the alignment loss does not bring them back. For these encoders the HCPE is warmed up first.
It is trained alone to reproduce the output of the encoder's original patch embedding (MSE between the two
`[B, H, W, C]` embeddings), so Stage 1 starts from tokens that are already close to the teacher's. The
transformer is not run during the warmup, so an epoch costs little more than reading the images.

The warmup is an option of Stage 1, off by default:

```yaml
training:
  hcpe_warmup:
    epochs: 2                # 0 = off; 2 epochs of ImageNet-1k at batch 256 = 10,008 updates
    batch_size: 256          # images per update, split evenly over the GPUs
    learning_rate: 1.0e-3    # AdamW, linear warmup, then cosine decay to 0
    lr_warmup_updates: 100
```

```bash
# any Stage 1 config, from the command line
GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml --set training.hcpe_warmup.epochs=2
```

- It runs once, at the start of a new Stage 1 run, before the first update of the schedule above. Only the
  HCPE changes; the decoder keeps its initialisation.
- Progress is printed and written to `hcpe_warmup.jsonl` (`relative_mse` = MSE divided by the power of the
  target embedding). The result is saved as `hcpe_warmup.pt` with `hcpe_warmup.json` in the run folder; a
  restart with the same settings loads it instead of training again.
- `--resume`, `--continue-from` and `--init-from` skip it: the HCPE then comes from the checkpoint.
- The first `alignment` value (1 − cos) that Stage 1 logs shows how close the warmed-up encoder is to the
  teacher.

`scripts/pretrain_hcpe_patchify.py` does the same fit offline on a cached image subset and writes a weight
file for `model.encoder.hcpe_init` (SigLIP2 configs); use one of the two, not both. On SigLIP2-L/16 this
offline fit (100k updates at batch 64) raised the cosine between the HCPE encoder's tokens and the teacher's
from 0.30 (random HCPE) to 0.98 on 1,000 validation images.

## SigLIP2 encoder (optional)

The same recipe runs on SigLIP2-L/16 (256 px) through `model.encoder.type: siglip2`. Its Stage 1 config
enables the 2-epoch HCPE warmup:

```bash
python scripts/prepare_siglip2.py --repo google/siglip2-large-patch16-256

GPUS=0,1,2,3 scripts/train.sh configs/siglip2/stage1_alignment.yaml    # HCPE warmup, then Stage 1
GPUS=0,1,2,3 scripts/train.sh configs/siglip2/stage2_gan.yaml
```

`configs/siglip2/stage1_no_hcpe.yaml` is the control without the HCPE: the original SigLIP2 encoder stays
frozen and only the decoder trains.

## Tests

```bash
pytest -q
```

The tests run on CPU with tiny models; no weights or data are needed.

## Third-party code

- `hae/models/efficientvit_compat.py` and the decoder in `hae/models/generator.py` derive from
  [EfficientViT / DC-AE](https://github.com/mit-han-lab/efficientvit) (Apache-2.0,
  `third_party/EfficientViT-LICENSE`).
- The discriminator backbone is [DINO](https://github.com/facebookresearch/dino) ViT-S/8 (Apache-2.0,
  `third_party/DINO-LICENSE`).
- [DINOv3](https://github.com/facebookresearch/dinov3) is not included; clone it into `vendor/dinov3` and follow
  its license.
