# PINO: DINO Foundation Model for Gaussian Splats

A research project aimed at building a native 3D Foundation Model for Gaussian Splats using a self-supervised DINO-style objective.

## Project Overview

Traditional approaches for learning on 3D Gaussian Splats rely heavily on either rasterizing the splats into 2D views or discretizing them into rigid voxel grids. 

This project discards legacy assumptions (like voxels) and treats Gaussian Splats natively as 3D tokens. By leveraging state-of-the-art unstructured point-cloud processing and foundation model techniques, we build robust representations that understand both local geometric structures and global semantic invariance.

### Architectural Decisions

1. **Architecture**: **Native 3D Transformer** operating directly on the splat tokens (similar to Point-MAE / Point-BERT).
2. **Patch Grouping**: **Farthest Point Sampling (FPS) + K-Nearest Neighbors (KNN)**. We use FPS to select uniformly distributed patch centers and KNN to group the $K$ closest splats into local patches.
3. **Feature Representation**: **Localized 14D Features**. Each splat is represented by 14 parameters: relative position (`mean - patch_center`), color (3), scale (3), quaternion (4), and opacity (1).
4. **Patch Processing**: **Point-MAE Style**. A local MLP and Max Pooling aggregate the $K$ splats into a single token embedding per patch before being passed into the main Transformer Encoder.
5. **Training Objective**: **Hybrid Masking + Augmentation DINO**. Uses random masking for local structural learning (MAE) and 3D geometric augmentations (rotation, jitter) for global semantic invariance.

## Progress Tracker

- [x] **Project Initialization**: Basic boilerplate setup (Hydra, Pydantic, uv).
- [x] **Architecture Design**: Defined the Native 3D DINO pipeline (FPS + KNN, Localized Features).
- [x] **Data Preprocessing**:
  - [x] Implement FPS and KNN for splats.
  - [x] Extract raw scales and quaternions from PLY.
  - [x] Generate localized 14D feature tensors.
- [x] **Transformer Architecture**:
  - [x] Implement Point-MAE style patch embedding (MLP + Pool).
  - [x] Implement standard Transformer Encoder.
- [x] **2D DINO Distillation** (first training objective):
  - [x] Rendered-view data pipeline (image + DINOv3 tokens + visible splats + splat->patch correspondence).
  - [x] Rendered-embedding alignment loss and per-splat point loss.
  - [x] Single-GPU training loop with validation and visualisations.
- [ ] **DINO Framework**:
  - [ ] Implement Teacher/Student paradigm.
  - [ ] Implement 3D augmentations (Geometric transforms).
  - [ ] Implement patch masking logic.
- [ ] **Training Loop**:
  - [ ] Setup DINO loss.
  - [ ] Multi-GPU / scaling support.

## 2D → 3D DINO Distillation

The first objective aligns per-splat embeddings with frozen DINOv3 (ViT-L/16) features of rendered views.

1. **Scenes** (`pino.data.scene`, `pino.data.dataset`): DataLoader workers read the uncompressed PLY of an
   InteriorGS scene directly into tensors and propose candidate cameras at eye height inside the room
   polygons of `structure.json`.
2. **Views** (`pino.data.views.ViewGenerator`, GPU): candidates are rendered with gsplat; views inside
   furniture, facing a wall or looking out of the scene are rejected. DINOv3 embeds the kept renders and the
   splats in each view frustum are cropped out. Each visible splat (depth tested against the render) gets the
   index of the DINO patch it lands in. The result is a `ViewBatch`: image, DINO tokens, splats, correspondence.
3. **Model** (`pino.model.SplatTransformer`): FPS centres with Voronoi grouping, a mini-PointNet on the
   localised 14D features, a transformer over the group tokens, and per-splat decoding from the point feature
   plus the interpolated nearby tokens.
4. **Loss** (`pino.losses`): per-splat embeddings are alpha-blended into the view with gsplat
   (differentiable), pooled onto the DINO patch grid, normalised by coverage, linearly projected and compared
   with the standardised DINO tokens by cosine distance. A second, optional term matches each visible splat
   directly to its patch token. The projection is linear, so it commutes with blending:
   `model.to_teacher_space(model.project(features))` gives each splat's embedding in DINO space without rendering.

```bash
uv run python -m pino.train                                # defaults in conf/config.yaml
uv run python -m pino.train hparams.max_steps=1000 data.scenes_per_batch=2
```

Requirements: the DINOv3 checkpoint in the local HF cache (`HF_HOME`), and nvcc + `CUDA_HOME` so gsplat can
JIT-compile its CUDA kernels on first use (cached afterwards). Outputs (metrics.jsonl, checkpoints,
`vis/val_*.jpg` showing render | DINO PCA | predicted PCA) go to the Hydra run dir.

`val/baseline/cos` is the score of predicting the mean DINO token; `val/render/cos_raw` must beat it.

**SplatFormer baseline.** `model.arch=splatformer` swaps the encoder for SplatFormer's Point Transformer V3
backbone (`pino.splatformer`, vendored PTv3 in `src/pino/third_party/pointcept/`). Every splat is a token,
views are min-max normalised and voxelised at 1/384, and attention runs over patches of 1024 splats along
space-filling curves that change from block to block, in a multi-scale U-Net. Data, losses and projection head are
shared with `SplatTransformer`. Needs `spconv-cu126` (installed by `uv sync`).

```bash
uv run python -m pino.train model.arch=splatformer
```

To inspect a checkpoint, colour every splat of a scene by the top-3 PCA components of its embedding and render it
(also writes a PCA-coloured `.ply`; `--viser` serves it interactively):

```bash
uv run python -m pino.pca_render --ckpt outputs/<run>/latest.pt --scene 0101_839973 --out pca_out
```

To measure how well the embeddings identify objects (linear probe over `labels.json` object classes on held-out
scenes, against raw-attribute and random-init baselines):

```bash
uv run python -m pino.object_probe --ckpt outputs/<run>/ckpt_030000.pt --out probe_out
```

Experiments, hypotheses and run summaries are tracked in [`lab_notebook/`](lab_notebook/README.md).

## Tech Stack

- **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management.
- **[Hydra](https://hydra.cc/)** for composing hierarchical configurations.
- **[Pydantic](https://docs.pydantic.dev/)** for robust config data validation.
- **[Pyrefly](https://pyrefly.org/)** for fast, incremental static type checking.
- **[PyTorch](https://pytorch.org/)** for deep learning operations.

## Installation

This project uses `uv`. To install the dependencies:

```bash
uv sync
```

## Running the Code

The splat viewer entry point is `src/pino/main.py`.

```bash
uv run python src/pino/main.py
```

You can override Hydra configurations from the command line:

```bash
uv run python src/pino/main.py hparams.learning_rate=0.005
```
