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
- [ ] **Data Preprocessing**:
  - [ ] Implement FPS and KNN for splats.
  - [ ] Extract raw scales and quaternions from PLY.
  - [ ] Generate localized 14D feature tensors.
- [ ] **Transformer Architecture**:
  - [ ] Implement Point-MAE style patch embedding (MLP + Pool).
  - [ ] Implement standard Transformer Encoder.
- [ ] **DINO Framework**:
  - [ ] Implement Teacher/Student paradigm.
  - [ ] Implement 3D augmentations (Geometric transforms).
  - [ ] Implement patch masking logic.
- [ ] **Training Loop**:
  - [ ] Setup DINO loss.
  - [ ] Multi-GPU / scaling support.

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

The main entry point is `src/main.py`.

```bash
uv run python src/main.py
```

You can override Hydra configurations from the command line:

```bash
uv run python src/main.py hparams.learning_rate=0.005
```
