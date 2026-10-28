# Vendored Point Transformer V3 (for the SplatFormer baseline)

Used by `pino.splatformer.SplatFormer`. Sources:

- [Pointcept](https://github.com/Pointcept/Pointcept), MIT (see `LICENSE`), at the fork SplatFormer pins as a
  submodule: `ChenYutongTHU/Pointcept@c4aa232bfbc24dc0f3d6699dff03632f9d0ff595`.
  - `ptv3.py` <- `pointcept/models/point_transformer_v3/point_transformer_v3m1_base.py`
  - `structure.py` <- `pointcept/models/utils/structure.py`
  - `modules.py` <- `pointcept/models/modules.py`
  - `misc.py` <- `pointcept/models/utils/misc.py`
  - `serialization/` <- `pointcept/models/utils/serialization/` (unchanged)
- [SplatFormer](https://github.com/ChenYutongTHU/SplatFormer) `@446ffb5`, `models/pointtransformer_v3.py`: its
  `PointTransformerV3` class (Pointcept's PT-v3m1 plus `embedding_type` and `turn_off_bn`) replaces Pointcept's
  in `ptv3.py`. The SplatFormer repo has no code licence file.

## Changes (each marked `PINO:` in the code)

1. Imports made package-relative; the Pointcept model registry decorator is dropped.
2. `flash_attn.flash_attn_varlen_qkvpacked_func` -> `varlen_qkvpacked_attention` (torch SDPA over the same
   packed patches; no flash-attn build exists for torch 2.12 / CUDA 13 here). Flash-attn ran in fp16; SDPA runs
   in the autocast dtype (bf16 in training).
3. `torch_scatter.segment_csr(src, indptr, reduce)` -> `torch.segment_reduce(src, reduce, offsets=indptr)`.
4. `timm.models.layers.DropPath` inlined.
5. SplatFormer's decoder container `PointSequential_intermediate_output` -> `PointSequential` (it only also
   returned per-stage intermediates that SplatFormer discards). PDNorm is not vendored (asserted off, as in
   SplatFormer's config).
6. `PointSequential` casts bf16 features to fp32 before spconv modules. spconv has no bf16 kernels; under
   autocast its autograd path casts features and weights to fp16, but its inference (no-grad) path casts nothing
   and needs features in the (fp32) weight dtype.
7. `Block.use_checkpoint`: optional activation checkpointing of each block (math unchanged; the RNG state is
   restored so DropPath masks match). A PINO batch is ~1.8M splats, which does not fit without it.

Runtime dependencies: `spconv-cu126` (sparse-conv positional encoding) and `addict`.
