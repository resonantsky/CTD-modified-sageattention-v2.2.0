import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .quant_per_block_varlen import per_block_int8 as per_block_int8_varlen
from .quant_per_block_hd96 import per_block_int8_hd96
from .attn_qk_int8_per_block_h96 import forward as attn_h96_false
from .attn_qk_int8_per_block_h96_causal import forward as attn_h96_true
from .attn_qk_int8_per_block import forward as attn_false
from .attn_qk_int8_per_block_causal import forward as attn_true
from .attn_qk_int8_block_varlen import forward as attn_false_varlen
from .attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from typing import Any, List, Literal, Optional, Tuple, Union

import time as _time
import sys as _sys
import atexit as _atexit
import os as _os

# --- SageAttention Console Logging ---
# Set _SAGE_LOG_ENABLED = False to disable all logging.
_SAGE_LOG_ENABLED = True
_SAGE_LOG_DIR  = r"E:\Sd.Next\sagebench"
_os.makedirs(_SAGE_LOG_DIR, exist_ok=True)
_SAGE_LOG_FILE = _os.path.join(_SAGE_LOG_DIR, "sage.log")
_SAGE_LOGGED_SHAPES = set()  # shapes written to sage.log this session
_SAGE_STEP_CALLS = 0
_SAGE_SEEN_PATHS = set()
_SAGE_LAST_TIME = 0.0
_SAGE_STEP_START = 0.0
_SAGE_SPIN_IDX = 0
_SAGE_LAST_SPIN = 0.0
_SAGE_TAG_WIDTH = 90      # fixed visible width — every write pads to this

# Braille spinner frames
_SAGE_FRAMES = "⠁⠂⠄⡀⢀⠠⠐⠈"

# ANSI colors — matched to the SD.Next console palette
_S = {
    'R':  '\033[0m',        # Reset
    'sp': '\033[95m',       # Bright Magenta  — spinner character
    'sg': '\033[92m',       # Bright Green    — "Sage Attention" & path
    'ct': '\033[93m',       # Bright Yellow   — call count
    'tm': '\033[96m',       # Bright Cyan     — timing
    'dt': '\033[90m',       # Grey            — separators
    'br': '\033[95m',       # Bright Magenta  — || bracket
}

def _sage_write(parts):
    """Write colored text padded to fixed width so each write fully overwrites the last."""
    colored = ""
    vis_len = 0
    for text, color in parts:
        colored += f"{color}{text}{_S['R']}" if color else text
        vis_len += len(text)
    pad = max(0, _SAGE_TAG_WIDTH - vis_len)
    _sys.stdout.write('\033[s' + colored + ' ' * pad + '\033[u')
    _sys.stdout.flush()

def _sage_log(func_name, q, k, v, tensor_layout, is_causal, dtype, path_label):
    """Animated spinner during attention computation. Appends inline after the
    progress bar text — the bar's next \\r update overwrites it cleanly.
    Fixed-width padding ensures no visual remnants."""
    if not _SAGE_LOG_ENABLED:
        return
    global _SAGE_STEP_CALLS, _SAGE_SEEN_PATHS
    global _SAGE_LAST_TIME, _SAGE_STEP_START
    global _SAGE_SPIN_IDX, _SAGE_LAST_SPIN
    now = _time.time()
    gap = now - _SAGE_LAST_TIME if _SAGE_LAST_TIME > 0 else 0

    # ── New step detected (>0.3s gap between call bursts) ──
    if gap > 0.3 and _SAGE_LAST_TIME > 0:
        # Reset for new step — fall through so the first call still renders
        _SAGE_STEP_CALLS = 0
        _SAGE_SEEN_PATHS = set()
        _SAGE_STEP_START = now
        _SAGE_LAST_SPIN = 0.0
    elif _SAGE_STEP_START == 0:
        _SAGE_STEP_START = now

    _SAGE_LAST_TIME = now
    _SAGE_STEP_CALLS += 1
    _SAGE_SEEN_PATHS.add(path_label)

    # ── Shape file logging (unique shapes only) ──
    global _SAGE_LOGGED_SHAPES
    if tensor_layout == "varlen":
        shape_key = f"varlen tokens={q.size(0)} heads={q.size(1)} head_dim={q.size(2)} causal={is_causal} dtype={dtype}"
    elif tensor_layout == "HND":
        shape_key = f"B={q.size(0)} heads={q.size(1)} seq={q.size(2)} head_dim={q.size(3)} layout=HND causal={is_causal} dtype={dtype}"
    else:  # NHD
        shape_key = f"B={q.size(0)} seq={q.size(1)} heads={q.size(2)} head_dim={q.size(3)} layout=NHD causal={is_causal} dtype={dtype}"
    if shape_key not in _SAGE_LOGGED_SHAPES:
        _SAGE_LOGGED_SHAPES.add(shape_key)
        with open(_SAGE_LOG_FILE, "a") as _f:
            _f.write(shape_key + "\n")

    # ── Live spinner (throttled to ~60ms) ──
    if now - _SAGE_LAST_SPIN > 0.06:
        _SAGE_LAST_SPIN = now
        # _SAGE_SPIN_IDX = (_SAGE_SPIN_IDX + 1) % len(_SAGE_FRAMES)  # spinner — hidden
        # frame = _SAGE_FRAMES[_SAGE_SPIN_IDX]                        # spinner — hidden
        # elapsed = now - _SAGE_STEP_START if _SAGE_STEP_START > 0 else 0  # hidden
        # Triton-style shape: B×H×N×D
        if tensor_layout == "varlen":
            # q is [total_tokens, H, D]
            shape_str = f"{q.size(0)}×{q.size(1)}×{q.size(2)}"
        elif tensor_layout == "HND":
            shape_str = f"{q.size(0)}×{q.size(1)}×{q.size(2)}×{q.size(3)}"
        else:  # NHD
            shape_str = f"{q.size(0)}×{q.size(1)}×{q.size(2)}×{q.size(3)}"
        # Active autotune config
        try:
            import sageattention.attn_qk_int8_per_block as _m
            _c = _m.configs[0]
            cfg_str = (f"BM{_c.kwargs['BLOCK_M']} BN{_c.kwargs['BLOCK_N']}"
                       f" nw{_c.num_warps} wpe{_c.kwargs['waves_per_eu']} ns{_c.num_stages}")
        except Exception:
            cfg_str = "?"
        # (" ", None), (frame, _S['sp']), ("  ", None),  # spinner — hidden
        # (" shp:", _S['dt']), (shape_str, _S['ct']),        # shape detail — hidden
        # (" cfg:", _S['dt']), (cfg_str, _S['tm']),      # cfg detail — hidden
        _sage_write([
            (" [sage attention]", _S['sg']),
            (" shp:", _S['dt']), (shape_str, _S['ct']),
            (" cfg:", _S['dt']), (cfg_str, _S['tm']),
        ])
# --- End Logging ---

def sageattn(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    tensor_layout: str ="HND", 
    is_causal=False, 
    sm_scale: Optional[float] = None, 
    smooth_k: bool =True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    """

    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim in [64, 96, 128], "headdim should be in [64, 96, 128]."

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.float().mean(dim=seq_dim, keepdim=True).to(k.dtype)
        k -= km
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if headdim == 96:
        q_int8, q_scale, k_int8, k_scale = per_block_int8_hd96(q, k, sm_scale=sm_scale, tensor_layout=tensor_layout)
        if is_causal:
            _sage_log("sageattn", q, k, v, tensor_layout, is_causal, dtype, "INT8-h96 causal")
            return attn_h96_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
        else:
            _sage_log("sageattn", q, k, v, tensor_layout, is_causal, dtype, "INT8-h96 non-causal")
            return attn_h96_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    import sageattention.attn_qk_int8_per_block as _sa_mod
    import sageattention.attn_qk_int8_per_block_causal as _sa_mod_causal
    _active_cfg = _sa_mod.configs[0] if _sa_mod.configs else None
    _BLKQ = _active_cfg.kwargs.get('BLOCK_M', 32) if _active_cfg else 32
    _BLKK = _active_cfg.kwargs.get('BLOCK_N', 16) if _active_cfg else 16

    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, BLKQ=_BLKQ, BLKK=_BLKK, sm_scale=sm_scale, tensor_layout=tensor_layout)

    if is_causal:
        _sage_log("sageattn", q, k, v, tensor_layout, is_causal, dtype, "INT8 causal")
        o = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)
    else:
        _sage_log("sageattn", q, k, v, tensor_layout, is_causal, dtype, "INT8 non-causal")
        o = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype)

    return o

def sageattn_varlen(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    is_causal: bool=False,
    sm_scale: Optional[float]=None, 
    smooth_k: bool=True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.
    
    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.
    
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    """
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device == cu_seqlens_q.device == cu_seqlens_k.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    assert cu_seqlens_q.dtype in [torch.int32, torch.int64] and cu_seqlens_k.dtype in [torch.int32, torch.int64], "cu_seqlens_q and cu_seqlens_k must have dtype torch.int32 or torch.int64."

    head_dim = q.size(-1)
    assert head_dim in [64, 128], "varlen only support head_dim [64, 128]."

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = k.float().mean(dim=0, keepdim=True).to(k.dtype)  # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k -= km

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        _sage_log("sageattn_varlen", q, k, v, "varlen", is_causal, dtype, f"INT8 varlen causal (max_q={max_seqlen_q}, max_k={max_seqlen_k})")
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        _sage_log("sageattn_varlen", q, k, v, "varlen", is_causal, dtype, f"INT8 varlen non-causal (max_q={max_seqlen_q}, max_k={max_seqlen_k})")
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    return o
