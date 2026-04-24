# SageAttention v2.2.0 - Windows ROCm 7.1 

SageAttention is a high-performance, low-bit attention module for PyTorch, optimized for AMD ROCm and RDNA architectures. It provides efficient quantized attention kernels and flexible integration for advanced transformer models.

This version has a static configuration module in \triton\config.py, in the case where head dimension is 64 or 128, blocks adapt to the fastest config, in this case 64:32 & 218:16.

Tuned for SD.Next on an RX6800 

## Directory Structure

- **core.py**  
  Main logic for quantized attention, kernel selection, and integration with Triton/ROCm. Imports and dispatches to specialized kernels and quantization routines.

- **quant.py**  
  Implements quantization routines and helpers, including per-block quantization and integration with fused CUDA/ROCm kernels.

- **triton/**  
  Contains all Triton-based kernel implementations and quantization helpers:
  - `attn_qk_int8_block_varlen.py`, `attn_qk_int8_per_block.py`, etc.:  
    Triton kernels for various quantized attention patterns (block, per-block, causal, varlen).
  - `quant_per_block.py`, `quant_per_thread.py`, etc.:  
    Quantization routines for different memory layouts and performance tradeoffs.
  - `config.py`:  
    Triton kernel launch configuration for ROCm/RDNA. This file defines:
    - Tile sizes (BLOCK_M, BLOCK_N)
    - Warp count (num_warps)
    - Pipeline depth (num_stages)
    - AMD-specific wavefront overlap hints (waves_per_eu)
    Edit these parameters to tune performance for your hardware. Changes take effect on next import. The file also suppresses noisy kernel/backend warnings on import, making logs cleaner during development and runtime.

- **\*.pyd files**  
  Compiled binary extensions for performance-critical routines (e.g., `_fused.cp312-win_amd64.pyd`, `_qattn_sm80.cp312-win_amd64.pyd`). These are required for fast inference and are loaded dynamically by the Python modules.

## Module Flow

1. **Import**:  
   `__init__.py` exposes main entry points from `core.py` for use in external projects.

2. **Kernel Selection**:  
   `core.py` detects available hardware and loads the appropriate binary extension or Triton kernel.

3. **Quantization & Attention**:  
   Quantization routines in `quant.py` and `triton/` prepare tensors for efficient attention computation.

4. **Execution**:  
   Attention is computed using the selected kernel (Triton or binary extension), with configuration from `triton/config.py`.

## Notes

- The `.pyd` files are essential for performance and must be included in any wheel or distribution.
- The package is designed for advanced users needing custom or high-performance attention on AMD GPUs.
