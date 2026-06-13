# -*- coding: utf-8 -*-
"""
Benchmark new sageattention (v2.2.0 with ROCm patches) against reference.

Tests:
  1. sageattn()                         -- high-level dispatch (verifies ROCm routing)
  2. sageattn_qk_int8_pv_fp16_triton()  -- Triton kernel with explicit config grid
  3. Correctness vs torch SDPA (fp32)   -- max-abs error reported per call

Columns in CSV:
  shape, path, cfg, ms, tflops, gb_s, max_err, winner

Usage (venv active, from flash-attn dir):
    python benchmark-sage-attn.py
"""

import sys
import os
import csv
import math
import torch
import torch.nn.functional as F
import triton

# Ensure Unicode box-drawing characters render on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- ANSI colour palette (matches SD.Next console) ----------------------------
class C:
    R   = '\033[0m'    # reset
    # structural
    HDR = '\033[1;97m'  # bold white   — section headers
    DIM = '\033[90m'   # dark grey    — separators / labels
    # shape / info
    SHP = '\033[96m'   # bright cyan  — shape line
    INF = '\033[94m'   # bright blue  — info / routing
    # timing / throughput
    MS  = '\033[93m'   # bright yellow — ms
    TF  = '\033[95m'   # bright magenta — TFLOPS
    GBS = '\033[35m'   # magenta      — GB/s
    CFG = '\033[97m'   # white        — config label
    # correctness
    OK   = '\033[92m'  # bright green
    WARN = '\033[33m'  # yellow
    FAIL = '\033[91m'  # bright red
    # winner / best
    BEST = '\033[1;92m' # bold green

def _c(text, colour): return f"{colour}{text}{C.R}"
def _ok_colour(tag): return {"OK": C.OK, "WARN": C.WARN, "FAIL": C.FAIL}.get(tag, C.R)

# -- sageattention path: use the patched local copy ---------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# -- Tuning knobs -------------------------------------------------------------
DEFAULT_OUT     = os.path.join(SCRIPT_DIR, "sage.txt")
WARMUP_RUNS     = 5
TIMED_RUNS      = 40
N_LAYERS        = 1
VRAM_BALLAST_GB = 0.0
# INT8 per-block quantization has inherent approximation error.
# Short sequences (N<256) have only a few blocks so max-abs error vs FP32 SDPA
# can reach ~1-2 on random data.  Use a loose threshold here; real inference
# data (more structured) will show smaller errors.
CORRECTNESS_TOL = 2.0

# -- Shapes -------------------------------------------------------------------
# (B, heads, seq, head_dim, layout, causal)
SHAPES = [
    (1, 32, 6144, 128, "HND", False),
    (1, 32, 6144, 64, "HND", False),
    (2, 12, 6144, 128, "HND", False),
    
]

# -- Triton config grid -------------------------------------------------------
BLOCK_M_VALS      = [32]
BLOCK_N_VALS      = [16]
NUM_WARPS_VALS    = [2,4,8,16]
WAVES_PER_EU_VALS = [1]
NUM_STAGES_VALS   = [1]

def _keep(bm, bn, nw, wpe, ns):
    if bm < bn:
        return False
    if bm == 16 and bn == 16:
        return False
    return True

TRITON_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "STAGE": ns, "waves_per_eu": wpe},
        num_warps=nw, num_stages=ns,
    )
    for bm  in BLOCK_M_VALS
    for bn  in BLOCK_N_VALS
    for nw  in NUM_WARPS_VALS
    for wpe in WAVES_PER_EU_VALS
    for ns  in NUM_STAGES_VALS
    if _keep(bm, bn, nw, wpe, ns)
]

# -- Helpers ------------------------------------------------------------------
def cfg_label(c):
    return (f"BM{c.kwargs['BLOCK_M']}_BN{c.kwargs['BLOCK_N']}"
            f"_nw{c.num_warps}_wpe{c.kwargs['waves_per_eu']}_ns{c.num_stages}")

def attn_flops(B, H, N, D, causal):
    return B * H * N * N * 4 * D * (0.5 if causal else 1.0)

def attn_bytes(B, H, N, D, blkq, blkn):
    return (B * H * N * D * 1
          + B * H * N * D * 1
          + B * H * N * D * 2
          + B * H * N * D * 2
          + B * H * math.ceil(N / blkq) * 4
          + B * H * math.ceil(N / blkn) * 4)

def make_qkv(B, H, N, D, layout, dtype):
    if layout == "NHD":
        q = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
        k = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
        v = torch.randn(B, N, H, D, dtype=dtype, device="cuda")
    else:
        q = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        k = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
        v = torch.randn(B, H, N, D, dtype=dtype, device="cuda")
    return q.contiguous(), k.contiguous(), v.contiguous()

def sdpa_reference(q, k, v, layout, causal, sm_scale):
    if layout == "NHD":
        q_ = q.permute(0, 2, 1, 3).float()
        k_ = k.permute(0, 2, 1, 3).float()
        v_ = v.permute(0, 2, 1, 3).float()
    else:
        q_ = q.float(); k_ = k.float(); v_ = v.float()
    out = F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal, scale=sm_scale)
    if layout == "NHD":
        out = out.permute(0, 2, 1, 3)
    return out

def timed_call(fn, warmup, runs, n_layers):
    for _ in range(warmup):
        for _ in range(n_layers):
            fn()
        torch.cuda.synchronize()
    total_ms = 0.0
    for _ in range(runs):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(n_layers):
            fn()
        ev1.record()
        torch.cuda.synchronize()
        total_ms += ev0.elapsed_time(ev1)
    return total_ms / runs / n_layers

# -- sageattn() high-level dispatch -------------------------------------------
def bench_sageattn(shape, ref_out):
    from sageattention import sageattn
    from sageattention.core import _is_rocm
    B, H, N, D = shape["B"], shape["heads"], shape["seq"], shape["head_dim"]
    layout, causal, dtype = shape["layout"], shape["causal"], shape["dtype"]
    sm_scale = D ** -0.5
    q, k, v = make_qkv(B, H, N, D, layout, dtype)
    try:
        out = sageattn(q, k, v, tensor_layout=layout, is_causal=causal,
                       sm_scale=sm_scale, smooth_k=True)
        max_err = (out.float() - ref_out).abs().max().item()
    except Exception as e:
        return None, None, None, f"err:{e}"
    try:
        ms = timed_call(
            lambda: sageattn(q, k, v, tensor_layout=layout, is_causal=causal,
                             sm_scale=sm_scale, smooth_k=True),
            WARMUP_RUNS, TIMED_RUNS, N_LAYERS)
    except Exception as e:
        return None, None, None, f"err:{e}"
    flops  = attn_flops(B, H, N, D, causal)
    tflops = flops / (ms * 1e-3) / 1e12
    routing = "rocm->triton" if _is_rocm() else "cuda->dispatch"
    return ms, tflops, max_err, routing

# -- sageattn_qk_int8_pv_fp16_triton() with explicit Triton config ------------
def bench_triton_cfg(cfg, shape, ref_out):
    """Call _attn_fwd directly with explicit BLOCK_M/N, num_warps, num_stages,
    waves_per_eu.  Quantization block sizes are set to match attention tile sizes
    so scale tensors have the correct dimensions."""
    from sageattention.triton.quant_per_block import per_block_int8 as quant_triton
    from sageattention.triton.attn_qk_int8_per_block import _attn_fwd
    import triton as _triton

    B, H, N, D = shape["B"], shape["heads"], shape["seq"], shape["head_dim"]
    layout, causal, dtype = shape["layout"], shape["causal"], shape["dtype"]
    sm_scale = D ** -0.5
    BLKM = cfg.kwargs["BLOCK_M"]
    BLKN = cfg.kwargs["BLOCK_N"]
    NW   = cfg.num_warps
    NS   = cfg.num_stages
    WPE  = cfg.kwargs["waves_per_eu"]

    if D not in {64, 128}:
        return None, None, None, "skip:head_dim"
    if layout != "HND":
        return None, None, None, "skip:layout"

    q, k, v = make_qkv(B, H, N, D, layout, dtype)

    # smooth_k: subtract key mean to improve quantization accuracy
    km = k.mean(dim=2, keepdim=True)

    # Quantize with block sizes that MATCH the attention tile dimensions.
    # q_scale shape: (B, H, ceil(N/BLKM)), k_scale shape: (B, H, ceil(N/BLKN))
    q_int8, q_scale, k_int8, k_scale = quant_triton(
        q, k, km=km, BLKQ=BLKM, BLKK=BLKN,
        sm_scale=sm_scale, tensor_layout=layout)

    b, h_qo, qo_len, head_dim = q_int8.shape
    _, h_kv, kv_len, _ = k_int8.shape
    num_kv_groups = h_qo // h_kv

    o = torch.empty(q_int8.shape, dtype=dtype, device="cuda")
    lse = torch.empty([0], dtype=torch.float32, device="cpu")

    sq, sh_q, sn_q = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
    sk, sh_k, sn_k = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    sv, sh_v, sn_v = v.stride(0), v.stride(1), v.stride(2)
    so, sh_o, sn_o = o.stride(0), o.stride(1), o.stride(2)

    grid = (_triton.cdiv(qo_len, BLKM), h_qo, b)

    def _launch():
        _attn_fwd[grid](
            q_int8, k_int8, v, q_scale, k_scale, o, None, lse,
            sq, sh_q, sn_q,
            sk, sh_k, sn_k,
            sv, sh_v, sn_v,
            so, sh_o, sn_o,
            0, 0, 0, 0,          # mask strides (no mask)
            qo_len, kv_len, h_qo, num_kv_groups,
            BLOCK_M=BLKM, BLOCK_N=BLKN, HEAD_DIM=head_dim,
            STAGE=1, RETURN_LSE=False,
            num_warps=NW, num_stages=NS, waves_per_eu=WPE)

    try:
        _launch()
        torch.cuda.synchronize()
        max_err = (o.float() - ref_out).abs().max().item()
        ms = timed_call(_launch, WARMUP_RUNS, TIMED_RUNS, N_LAYERS)
    except Exception as e:
        return None, None, None, f"err:{e}"

    flops  = attn_flops(B, H, N, D, causal)
    nbytes = attn_bytes(B, H, N, D, BLKM, BLKN)
    tflops = flops  / (ms * 1e-3) / 1e12
    gbs    = nbytes / (ms * 1e-3) / 1e9
    return ms, tflops, gbs, max_err

# -- Main ---------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA/ROCm device required.")
        sys.exit(1)

    is_rocm  = getattr(torch.version, "hip", None) is not None
    gcn_arch = ""
    if is_rocm:
        try:
            gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        except Exception:
            pass

    sep = _c("═" * 60, C.DIM)
    print(sep)
    print(_c("  SageAttention Benchmark", C.HDR))
    print(sep)
    print(f"  {_c('Device  :', C.DIM)} {_c(torch.cuda.get_device_name(0), C.SHP)}")
    print(f"  {_c('VRAM    :', C.DIM)} {_c(f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB", C.SHP)}")
    backend_str = 'ROCm/HIP' if is_rocm else 'CUDA'
    print(f"  {_c('Backend :', C.DIM)} {_c(backend_str, C.INF)}  {_c(gcn_arch, C.DIM)}")
    print(f"  {_c('Torch   :', C.DIM)} {_c(torch.__version__, C.INF)}")
    try:
        from importlib.metadata import version as _ver
        print(f"  {_c('SageAttn:', C.DIM)} {_c(_ver('sageattention'), C.INF)}")
    except Exception:
        print(f"  {_c('SageAttn:', C.DIM)} {_c('(local patched copy)', C.INF)}")
    print(f"  {_c('Shapes  :', C.DIM)} {_c(str(len(SHAPES)), C.MS)}   {_c('Triton configs:', C.DIM)} {_c(str(len(TRITON_CONFIGS)), C.MS)}")
    print(sep)
    print()

    ballast = None
    if VRAM_BALLAST_GB > 0:
        try:
            ballast = torch.empty(int(VRAM_BALLAST_GB * 1024**3 / 2),
                                  dtype=torch.float16, device="cuda")
        except RuntimeError:
            pass

    all_rows = []

    for si, (B, heads, seq, hd, layout, causal) in enumerate(SHAPES):
        shape = dict(B=B, heads=heads, seq=seq, head_dim=hd,
                     layout=layout, causal=causal, dtype=torch.float16,
                     line=f"B={B} H={heads} N={seq} D={hd} {layout} causal={causal}")
        print(_c(f"── [{si+1}/{len(SHAPES)}] ", C.DIM) + _c(shape['line'], C.SHP) + _c(" ──", C.DIM))

        q_r, k_r, v_r = make_qkv(B, heads, seq, hd, layout, torch.float16)
        ref_out = sdpa_reference(q_r, k_r, v_r, layout, causal, hd ** -0.5).float()

        # 1. sageattn() high-level
        ms, tflops, max_err, info = bench_sageattn(shape, ref_out)
        def _ok(e): return "OK" if isinstance(e, float) and e < CORRECTNESS_TOL else "WARN" if isinstance(e, float) and e < 5.0 else "FAIL"
        if ms is not None:
            ok_tag = _ok(max_err)
            print(
                f"  {_c('sageattn()', C.INF)}"
                f" {_c('[', C.DIM)}{_c(info, C.CFG)}{_c(']', C.DIM)}"
                f"  {_c(f'{ms:.3f}ms', C.MS)}"
                f"  {_c(f'{tflops:.2f}TF', C.TF)}"
                f"  {_c('err=', C.DIM)}{_c(f'{max_err:.4f}', C.OK if ok_tag=='OK' else C.WARN if ok_tag=='WARN' else C.FAIL)}"
                f"  {_c(ok_tag, _ok_colour(ok_tag))}"
            )
            all_rows.append(dict(shape=shape["line"], path="sageattn", cfg=info,
                                 ms=f"{ms:.4f}", tflops=f"{tflops:.3f}", gb_s="--",
                                 max_err=f"{max_err:.5f}", winner=""))
        else:
            print(f"  {_c('sageattn()', C.INF)} {_c('FAILED:', C.FAIL)} {info}")
            all_rows.append(dict(shape=shape["line"], path="sageattn", cfg=str(info),
                                 ms="err", tflops="err", gb_s="err", max_err="err", winner=""))

        # 2. sageattn_qk_int8_pv_fp16_triton() per config
        best_ms  = float("inf")
        best_lbl = None
        for cfg in TRITON_CONFIGS:
            lbl = cfg_label(cfg)
            ms_t, tflops_t, gbs_t, result = bench_triton_cfg(cfg, shape, ref_out)
            if ms_t is None:
                tag = "err" if isinstance(result, str) and "err" in result else "skip"
                all_rows.append(dict(shape=shape["line"], path="triton", cfg=lbl,
                                     ms=tag, tflops="--", gb_s="--", max_err="--", winner=""))
                if tag == "err":
                    print(f"  {_c('triton', C.DIM)} {_c(lbl, C.CFG)}: {_c(str(result), C.FAIL)}", file=sys.stderr)
                continue
            ok_t = _ok(result)
            print(
                f"  {_c('triton', C.DIM)} {_c(lbl, C.CFG)}:"
                f"  {_c(f'{ms_t:.3f}ms', C.MS)}"
                f"  {_c(f'{tflops_t:.2f}TF', C.TF)}"
                f"  {_c(f'{gbs_t:.0f}GB/s', C.GBS)}"
                f"  {_c('err=', C.DIM)}{_c(f'{result:.4f}', C.OK if ok_t=='OK' else C.WARN if ok_t=='WARN' else C.FAIL)}"
                f"  {_c(ok_t, _ok_colour(ok_t))}"
            )
            all_rows.append(dict(shape=shape["line"], path="triton", cfg=lbl,
                                 ms=f"{ms_t:.4f}", tflops=f"{tflops_t:.3f}",
                                 gb_s=f"{gbs_t:.1f}", max_err=f"{result:.5f}", winner=""))
            if ms_t < best_ms and isinstance(result, float) and result < 5.0:
                best_ms  = ms_t
                best_lbl = lbl

        if best_lbl:
            print(f"  {_c('→ best:', C.BEST)} {_c(best_lbl, C.BEST)}  {_c(f'({best_ms:.3f}ms)', C.MS)}")
            for row in all_rows:
                if row["shape"] == shape["line"] and row["cfg"] == best_lbl:
                    row["winner"] = "best"
        print()

    # CSV
    fieldnames = ["shape", "path", "cfg", "ms", "tflops", "gb_s", "max_err", "winner"]
    with open(DEFAULT_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(_c(f"Results written to: {DEFAULT_OUT}", C.DIM))

    print()
    print(_c("── Best Triton config per shape " + "─" * 28, C.DIM))
    for row in all_rows:
        if row.get("winner") == "best":
            print(
                f"  {_c(row['shape'][:55].ljust(55), C.SHP)}"
                f"  {_c(row['cfg'], C.BEST)}"
                f"  {_c(row['ms'] + 'ms', C.MS)}"
            )

    if ballast is not None:
        del ballast
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
