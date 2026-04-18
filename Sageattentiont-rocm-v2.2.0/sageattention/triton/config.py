"""
Triton kernel launch configuration for sageattention.

Values are auto-detected at import time based on hardware.
Edit this file to override — changes take effect on the next import (restart/reload).

Parameters:
  BLOCK_M              – query tile size.  Must match the BLKQ passed to per_block_int8().
  BLOCK_N              – key/value tile size.  Must match BLKK passed to per_block_int8().
  num_warps_d64        – warp count when head_dim == 64
  num_warps_d128       – warp count when head_dim != 64  (e.g. 128, 256)
  num_stages_fwd       – Triton pipeline depth for non-causal forward (d64; d128 adds +1 on NVIDIA)
  num_stages_causal    – Triton pipeline depth for causal forward
  waves_per_eu         – AMD wavefront-overlap hint (0 = disabled / not passed to kernel)
"""

import warnings as _warnings
import torch as _torch

# Suppress noisy but harmless kernel/backend warnings that fire on every import
# or first kernel compilation, especially on ROCm where some CUDA paths are absent.
_warnings.filterwarnings("ignore", message=".*memory efficient attention.*", category=UserWarning)
_warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning)
_warnings.filterwarnings("ignore", message=".*operator.*does not have a kernel.*", category=UserWarning)
_warnings.filterwarnings("ignore", message=".*triton.*", category=UserWarning)
_warnings.filterwarnings("ignore", category=UserWarning, module="triton")

# ── hardware detection ────────────────────────────────────────────────────────
IS_ROCM: bool = getattr(_torch.version, "hip", None) is not None

def _is_rdna() -> bool:
    if not IS_ROCM:
        return False
    try:
        return _torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1")
    except Exception:
        return True   # assume RDNA on any ROCm device if query fails

IS_RDNA: bool = _is_rdna()

# ── launch parameters ─────────────────────────────────────────────────────────
if IS_ROCM:
    # ROCm / RDNA — benchmark winner: BM=32, BN=16, nw=2, wpe=3, ns=1
    BLOCK_M:           int = 32
    BLOCK_N:           int = 16
    num_warps_d64:     int = 2
    num_warps_d128:    int = 2
    num_stages_fwd:    int = 1
    num_stages_causal: int = 1
    waves_per_eu:      int = 3
else:
    # NVIDIA CUDA — upstream defaults
    BLOCK_M:           int = 128
    BLOCK_N:           int = 64
    num_warps_d64:     int = 4
    num_warps_d128:    int = 8
    num_stages_fwd:    int = 3   # d64; d128 uses +1 — handled in get_launch_params()
    num_stages_causal: int = 4
    waves_per_eu:      int = 0   # not passed to kernel when 0


# ── helper ────────────────────────────────────────────────────────────────────
def get_launch_params(head_dim: int, causal: bool) -> dict:
    """Return the Triton **kwargs for a kernel launch (num_warps, num_stages, waves_per_eu).

    Args:
        head_dim: attention head dimension (64, 128, …)
        causal:   True for causal kernels, False for non-causal
    """
    nw = num_warps_d64 if head_dim == 64 else num_warps_d128

    if IS_ROCM:
        # same pipeline depth for both causal and non-causal on AMD
        ns = num_stages_causal if causal else num_stages_fwd
    else:
        if causal:
            ns = num_stages_causal
        else:
            # NVIDIA non-causal: d128 benefits from one extra stage
            ns = num_stages_fwd if head_dim == 64 else num_stages_fwd + 1

    params = {"num_warps": nw, "num_stages": ns}
    if waves_per_eu:
        params["waves_per_eu"] = waves_per_eu
    return params
