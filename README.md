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
- [ ] **Transformer Architecture**:
  - [x] Implement Point-MAE style patch embedding (MLP + Pool).
  - [x] Implement standard Transformer Encoder.
- [x] **DINO Framework**:
  - [x] Implement Teacher/Student paradigm.
  - [x] Implement 3D augmentations (Geometric transforms).
  - [x] Implement patch masking logic.
- [x] **Training Loop**:
  - [x] Setup DINO loss.
  - [x] Multi-GPU / scaling support.

## Tech Stack

- **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management.
- **[Hydra](https://hydra.cc/)** for composing hierarchical configurations.
- **[Pydantic](https://docs.pydantic.dev/)** for robust config data validation.
- **[Weights & Biases](https://wandb.ai/)** for comprehensive training metrics logging.
- **[Pyrefly](https://pyrefly.org/)** for fast, incremental static type checking.
- **[PyTorch](https://pytorch.org/)** for deep learning operations.

## Installation

This project uses `uv`. To install the dependencies:

```bash
uv sync
```

## Training & Data Pipeline

The pipeline uses a self-supervised DINO framework directly on Gaussian Splats. No ground truth labels are required.

### 1. Data Preparation

Configure your `.env` file with the dataset paths:
```bash
INTERIOR_GS_PATH="/z/dat/InteriorGS"
PROCESSED_GS_PATH="/z/dat/ProcessedGS"
WANDB_API_KEY="your_api_key_here"
```

Because Farthest Point Sampling (FPS) and K-Nearest Neighbors (KNN) are computationally heavy, it is highly recommended to run the offline preprocessing script first. This crunches through the `.ply` files using multiprocessing and saves `.pt` feature tensors for instantaneous GPU loading.

```bash
uv run python src/pino/preprocess_dataset.py
```

### 2. Training & Logging

The project uses [Weights & Biases (Wandb)](https://wandb.ai/) for tracking training loss curves and learning rates. You can configure wandb logging in `conf/config.yaml` (`wandb.enabled=True`).

To launch the DINO Teacher/Student training loop over the processed dataset:

```bash
uv run python src/pino/main.py
```

You can configure `preprocess.on_the_fly=True` in `conf/config.yaml` to skip the offline step, but this will bottleneck training.

You can override Hydra configurations from the command line:

```bash
uv run python src/main.py hparams.learning_rate=0.005
```
