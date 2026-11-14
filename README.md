# Hyperspherical Autoencoder for High-Fidelity Image Reconstruction and Generation

[![arXiv](https://img.shields.io/badge/arXiv-2601.22904-b31b1b.svg)](https://arxiv.org/abs/2601.22904)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HAE-yellow)](https://huggingface.co/wkdgnsgo/hae)

<!-- [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dino%20SAE-yellow)](https://huggingface.co/wkdgnsgo/dino-sae) -->

This repository is the official implementation of the paper: **"Hyperspherical Autoencoder for High-Fidelity Image Reconstruction and Generation"**.

---

![Teaser Image](./assets/intro.jpg)

- **Figure 1.** Generated images from LightningDiT-XL from HAE latent\*

---

<!-- ## 🔥 News

| Date           | Description                                            |
| :------------- | :----------------------------------------------------- |
| **2026.02.14** | Released SAE checkpoints and DDT-XL (80 epochs) model! |
| **2026.02.05** | Training in progress... Code coming soon!              |
| **2026.01.30** | Paper available on arXiv.                              | -->

## 📖 Abstract

Recent studies have explored using pretrained Vision Foundation Models (VFMs)
such as DINO for generative autoencoders, showing strong generative performance.
Unfortunately, existing approaches often suffer from limited reconstruction fidelity
due to the loss of high-frequency details. In this work, we present the Hyper-
spherical Autoencoder (HAE), a framework that bridges semantic representation
and pixel-level reconstruction. Our key insight is that while semantic information
in contrastive representations is primarily directional, enforcing strict magnitude
matching hinders the preservation of fine-grained details. To address this, we
introduce a Directional Feature Alignment objective that enforces semantic consis-
tency while allowing flexible feature magnitudes for detail retention, alongside a
Hierarchical Convolutional Patch Embedding module to enhance local structure
preservation. Furthermore, observing that SSL-based representations intrinsically
lie on a hypersphere, we employ Riemannian Flow Matching to train a Diffu-
sion Transformer (DiT) directly on this spherical latent manifold. Notably, our
manifold-aware DiT exhibits highly efficient convergence, achieving an exceptional
gFID of 1.96 alongside a reconstruction rFID of 0.78 and a PSNR of 25.2 dB,
validating the advantages of our manifold-aware approach. Co

## 🗂️ Repository

| Path | Content |
| :--- | :--- |
| [`stage1/`](./stage1) | Autoencoder (HAE) training and evaluation: HCPE + directional feature alignment, then DinoDisc GAN |
| [`src/efficientvit/`](./src/efficientvit) | DC-AE / EfficientViT components |

## 🤗 Pretrained Weights

Weights are hosted on Hugging Face: [wkdgnsgo/hae](https://huggingface.co/wkdgnsgo/hae).

| File | Content |
| :--- | :--- |
| `hae_no_smoothing.pt`, `hae_smoothing.pt` | HAE autoencoder (DINOv3 ViT-L/16 + HCPE + decoder), without and with smoothing |
| `0000550_ema0995.pt`, `0001000_qknorm_ema.pt` | Generation model (LightningDiT) checkpoints |
| `stats_all_element.pt` | Latent mean and std per token and channel (256 × 1024) |

The autoencoder files load directly with the code in `stage1/`; see
[Pretrained weights](./stage1/README.md#pretrained-weights).

## 🏋️ Autoencoder Training (`stage1/`)

The autoencoder is trained in two steps. Setup, data layout, evaluation and every option are described in
[`stage1/README.md`](./stage1/README.md).

```bash
cd stage1
pip install -r requirements.txt
git clone https://github.com/facebookresearch/dinov3.git vendor/dinov3
python scripts/prepare_perceptual.py --fid

# Stage 1: HCPE + decoder, L1 + LPIPS + directional (cosine) alignment, 25 epochs
GPUS=0,1,2,3 scripts/train.sh configs/stage1_alignment.yaml

# Stage 2: decoder-only DinoDisc GAN, 10 epochs, from the EMA weights of Stage 1
GPUS=0,1,2,3 scripts/train.sh configs/stage2_gan.yaml
```

| Step | Trains | Learning rate | Adam betas | Length |
| :--- | :--- | :--- | :--- | :--- |
| Stage 1, epochs 0 – 3 | HCPE + decoder | 1e-5 | (0.9, 0.999) | 3 epochs |
| Stage 1, epochs 3 – 25 | HCPE + decoder | 1e-4 | (0.9, 0.9) | 22 epochs |
| Stage 2 (GAN) | decoder | 5.4e-5 (G and D) | (0.5, 0.9) | 10 epochs |

Global batch 512 on ImageNet-1k at 256 × 256; EMA decay 0.999 (Stage 1) and 0.9975 (Stage 2).

**HCPE warmup (optional).** Encoders that are sensitive to their patch embedding, such as SigLIP2 or DINOv2,
need the HCPE to be warmed up before Stage 1: it is first fitted to the output of the encoder's original patch
embedding for about 2 epochs (10k updates at batch 256). Enable it with
`--set training.hcpe_warmup.epochs=2`; see [HCPE warmup](./stage1/README.md#hcpe-warmup).

<!--
**DINO-SAE** bridges semantic representation and pixel-level reconstruction through three key contributions:

1.  **Hierarchical Convolutional Patch Embedding**: Enhances local structure and texture preservation, mitigating the information bottleneck of standard ViT patchification.
2.  **Cosine Similarity Alignment**: Enforces semantic consistency (direction) while allowing flexible feature magnitudes for detail retention, resolving the conflict between semantic preservation and reconstruction.
3.  **Riemannian Flow Matching (RFM)**: Trains a Diffusion Transformer (DiT) directly on the spherical latent manifold, leveraging the intrinsic geometry of SSL-based representations.

Experiments on ImageNet-1K demonstrate that our approach achieves **0.37 rFID** and **26.2 dB PSNR**, setting a new state-of-the-art for VFM-based autoencoders.

## 🚀 Key Features

- **High-Fidelity Reconstruction**: Outperforms RAE and VAVAE in pixel-level reconstruction quality.
- **Efficient Convergence**: Riemannian Flow Matching accelerates training, achieving a gFID of 3.47 at just 80 epochs (6.67x faster than SD-VAE baselines).
- **Scale-Decoupled Training**: Uses Cosine Similarity Loss to decouple semantic alignment from feature magnitude.

## 🛠️ Installation

### 1. Environment Setup

First, create the conda environment and install dependencies.

```bash
conda create -n dinosae python=3.11
conda activate dinosae

# 2. Install PyTorch first (Recommended for CUDA compatibility)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare DINOv3

1. **Clone the repository:**
   ```bash
   cd src
   git clone https://github.com/facebookresearch/dinov3.git
   ```

Download dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth from [Here](https://github.com/facebookresearch/dinov3)

### 3. Directory Structure

Ensure your project directory matches the following structure:

```text
DinoSae/
├── src/
│   ├── dinov3/
│   │   ├── dinov3/
│   │   └── ...
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth <-- Pretrained Weight
├── requirements.txt
└── README.md
```

## 🏗️ Architecture & Method

DINO-SAE modifies the standard ViT encoder to bridge the gap between semantic representation and high-fidelity reconstruction.

1.  **Hierarchical Convolutional Stem**: We replace the standard linear patch embedding with a 4-stage CNN. This preserves high-frequency details (edges, textures) that are typically lost during aggressive patchification.
2.  **Cosine Similarity Alignment**: Instead of MSE, we align features using Cosine Similarity. This decouples semantic alignment (direction) from signal strength (magnitude), resolving the optimization conflict between the frozen teacher (DINOv3) and the reconstruction objective.
3.  **Spherical Latent Manifold**: We utilize **Riemannian Flow Matching (RFM)** on a hyperspherical latent space, leveraging the observation that SSL features naturally lie on a hypersphere.

## 💻 Usage

### Data Preparation

### Training (Autoencoder)

We employ a **4-Stage Progressive Training Strategy** to balance semantic alignment and reconstruction quality.

**Stage 1: Semantic-Structural Alignment**
Trains the Convolutional Patch Embedding and Decoder from scratch.

<!-- ```bash
python train_ae.py \
    --stage 1 \
    --data_path /path/to/imagenet \
    --batch_size 256 \
    --lr 1e-5 \
    --lambda_cos 0.5 \
    <!-- --lambda_lpips 1.0
``` -->

<!-- **Stage 2: Adversarial Adaptation** Adds GAN loss with a DINO-based Discriminator to enhance textural realism. -->

<!-- ```bash
python train_ae.py \
    --stage 2 \
    --resume output/stage1/checkpoint.pth \
    --lr 1e-4 \
    --lambda_adv 0.5
``` -->

<!-- **Stage 3: Decoder Refinement** Freezes the entire encoder (including the patch embedding) to fix the semantic latent space, fine-tuning only the decoder. -->

<!-- ```bash
python train_ae.py \
    --stage 3 \
    --resume output/stage2/checkpoint.pth \
    --freeze_encoder \
    --lr 1e-4
``` -->

<!-- **Stage 4: Noise Augmentation** Injects noise into the latents while keeping the encoder frozen to make the decoder robust for generative modeling. -->

<!-- ```bash
python train_ae.py \
    --stage 4 \
    --resume output/stage3/checkpoint.pth \
    --noise_aug True \
    --lr 5.4e-5
``` -->

<!-- **Generation (Riemannian Flow Matching)**
Train the Diffusion Transformer (DiT) on the spherical latent space extracted by DINO-SAE. -->

<!--
```bash
python train_dit.py \
    --ae_checkpoint output/stage4/checkpoint.pth \
    --model DiT-XL/2 \
    --rfm True \
    --epochs 80
``` -->

<!-- ## 📊 Results

### Reconstruction Quality (ImageNet 256x256)

DINO-SAE significantly outperforms existing VFM-based autoencoders in terms of pixel-level fidelity.

| Model               |  rFID ↓  |  PSNR ↑   |
| :------------------ | :------: | :-------: |
| SD-VAE              |   0.62   |   26.04   |
| VAVAE               |   0.28   |   27.96   |
| RAE                 |   0.59   |   18.94   |
| **DINO-SAE (Ours)** | **0.37** | **26.20** | -->

<!-- ### Generation Quality

Using DINO-SAE latents leads to faster convergence and higher generation quality compared to RAE and VAVAE baselines.

| Method                         | Epochs | gFID ↓ |   IS ↑    |
| :----------------------------- | :----: | :----: | :-------: |
| RAE + LightningDiT-XL          |   80   |  4.28  |     -     |
| RAE + DiTDH-XL                 |   80   |  2.16  |   214.8   |
| **DINO-SAE + LightningDiT-XL** |   80   |  3.47  | **202.1** |
| **DINO-SAE + DiTDH-XL**        |   80   |  2.66  |   209.7   |

## 📜 Citation

If you find this work useful for your research, please cite our paper: --> -->

<!-- ```bibtex
@article{dino_sae_2025,
  title={DINO-SAE: DINO Spherical Autoencoder for High-Fidelity Image Reconstruction and Generation},
  author={Anonymous Authors},
  journal={Under Review at ICML},
  year={2025}
}
``` -->

<!-- ## 🙏 Acknowledgement

This project is built upon the following open-source projects. We thank the authors for making their code and models available.

<!--
- [cite_start]**[DINOv3](https://github.com/facebookresearch/dinov2)**: We utilize the pre-trained DINOv3 ViT-Large as our frozen encoder backbone[cite: 265, 328].
- [cite_start]**[RAE](https://github.com/imzhengboy/RAE)**: Our noise augmentation strategy and baseline comparisons are adopted from the RAE codebase[cite: 447].
- [cite_start]**[DC-AE](https://arxiv.org/abs/2410.10733)**: We adopt the lightweight decoder architecture proposed in DC-AE for efficient upsampling[cite: 329].
- [cite_start]**[SiT](https://github.com/willmcneil/SiT)** & **[DiT](https://github.com/facebookresearch/DiT)**: The diffusion transformer architectures for our generative experiments are based on these repositories[cite: 128, 579]. -->

`````

``` -->
```` -->
`````
