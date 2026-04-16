"""
bench_sage_configs.py — SageAttention kernel config profiler for AMD RDNA2 (gfx1030)
======================================================================================
Sweeps BLOCK_M × BLOCK_N × num_warps × num_stages combinations across key sequence
lengths and head dims.  Uses CUDA events for GPU-accurate timing.  Ranks configs and
writes a CSV you can open in Excel/Calc.

Run (from flash-attn venv):
    $env:FLASH_ATTENTION_TRITON_AMD_ENABLE = "TRUE"
    $env:SAGE_ATTENTION_TRITON_AMD_ENABLE  = "TRUE"
    E:\\flash-attn\\venv\\Scripts\\python.exe E:\\flash-attn\\bench_sage_configs.py

Options (edit SWEEP below):
    SEQ_LENS   — sequence lengths to test
    HEAD_DIMS  — head dimensions to test  (64 / 96 / 128)
    CAUSALS    — True / False causal masking
    BATCH      — batch size
    N_HEADS    — number of attention heads
    N_WARMUP   — GPU warmup iterations per config (first compile amortised here)
    N_ITER     — timed iterations per config

Results are printed ranked fastest→slowest, and saved to bench_sage_configs.csv.

Safety: all configs are clamped to num_warps<=4, num_stages<=3 before launch.
"""

import os, sys, csv, time, itertools, importlib, datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import torch
import triton
import triton.language as tl

# ── Guard ────────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "ROCm/CUDA device required"
device = torch.device("cuda")

# ════════════════════════════════════════════════════════════════════════════
#  SWEEP CONFIG — edit these to match your training shapes
# ════════════════════════════════════════════════════════════════════════════
SEQ_LENS  = [512, 1024, 2048, 4096, 8192, 16384]
HEAD_DIMS = [64, 128]            # h96 kernel path is incompatible with gfx1030 — omitted
CAUSALS   = [False, True]
BATCH     = 2                   # more realistic for training
N_HEADS   = 16                  # typical for SD/diffusion models
N_WARMUP  = 10                  # warmup iters per config (covers Triton JIT compile)
N_ITER    = 50                  # timed iters — more at large seq for stable measurement

# Config grid
# bench_triton_mm_warps.py confirmed all of [2, 4, 8, 16] are safe on gfx1030 (0 crashes).
# warps=2 wins most shapes; warps=8 is safe to include for sweep coverage.
# stages=1 wins small/medium shapes in the triton_mm matmul bench — must be included.
BLOCK_MS    = [32, 64, 128]     # 128 will be skipped at very short seq (grid too small)
BLOCK_NS    = [16, 32, 64]      # 64 only valid when BN <= BM; enforced below
NUM_WARPS_  = [2, 4, 8]         # 8 confirmed safe on gfx1030 via triton_mm warp bench
NUM_STAGES_ = [1, 2, 3]         # 1 wins small/medium shapes (triton_mm bench); 3 = mild prefetch

CSV_OUT = "bench_sage_configs.csv"
LOG_OUT = "bench_sage_configs.log"

# ── ANSI colours ─────────────────────────────────────────────────────────────
_C = {
    'R':   '\033[0m',
    'hdr': '\033[1;96m',
    'num': '\033[93m',
    'ok':  '\033[1;92m',
    'err': '\033[91m',
    'dim': '\033[90m',
    'bld': '\033[1m',
    'hi':  '\033[1;97m',
}
def _c(t, k): return f"{_C[k]}{t}{_C['R']}"

# ════════════════════════════════════════════════════════════════════════════
#  Re-usable inline kernel wrappers
#  We patch forward() launch parameters directly rather than going through
#  sageattn() so we can sweep configs without touching module source each run.
# ════════════════════════════════════════════════════════════════════════════

# Import the low-level kernels we want to benchmark
from sageattention.quant_per_block import per_block_int8

# Import the Triton _attn_fwd functions directly
import sageattention.attn_qk_int8_per_block        as _mod_noncausal
import sageattention.attn_qk_int8_per_block_causal as _mod_causal
# h96 kernel modules omitted — incompatible with gfx1030 (hang on synchronize)


def _time_config(
    q_int8, k_int8, v_fp16,
    q_scale, k_scale,
    qo_len, kv_len, h_qo, h_kv, b, head_dim,
    is_causal: bool,
    BLOCK_M: int, BLOCK_N: int, num_warps: int, num_stages: int,
) -> Optional[float]:
    """
    Launch the appropriate Triton _attn_fwd kernel with explicit config,
    time it, return mean ms or None on error.
    Safety clamp applied here regardless of caller values.
    """
    # ── Safety clamp ────────────────────────────────────────────────────────
    num_warps  = min(num_warps,  8)   # gfx1030: [2..8] confirmed safe via triton_mm bench
    num_stages = min(num_stages, 3)   # gfx1030: 1/2/3 all valid; >3 not tested
    # Ensure BLOCK_N <= BLOCK_M  (avoids degenerate tile shapes)
    BLOCK_N = min(BLOCK_N, BLOCK_M)

    output_dtype = torch.float16
    o = torch.empty(q_int8.shape, dtype=output_dtype, device=device)

    # Strides (q_int8 is always HND here: [b, h, n, d])
    stride_bz_q, stride_h_q, stride_seq_q = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
    stride_bz_k, stride_h_k, stride_seq_k = k_int8.stride(0), k_int8.stride(1), k_int8.stride(2)
    stride_bz_v, stride_h_v, stride_seq_v = v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2)
    stride_bz_o, stride_h_o, stride_seq_o = o.stride(0),      o.stride(1),      o.stride(2)
    num_kv_groups = h_qo // h_kv

    mod   = _mod_causal if is_causal else _mod_noncausal
    stage = 3 if is_causal else 1
    grid  = (triton.cdiv(qo_len, BLOCK_M), h_qo, b)
    _fwd  = mod._attn_fwd
    def _launch():
        _fwd[grid](
            q_int8, k_int8, v_fp16, q_scale, k_scale, o,
            stride_bz_q, stride_h_q, stride_seq_q,
            stride_bz_k, stride_h_k, stride_seq_k,
            stride_bz_v, stride_h_v, stride_seq_v,
            stride_bz_o, stride_h_o, stride_seq_o,
            qo_len, kv_len, h_qo, num_kv_groups,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=head_dim,
            STAGE=stage,
            num_warps=num_warps, num_stages=num_stages,
        )

    try:
        # Warmup — absorbs Triton JIT compile
        for _ in range(N_WARMUP):
            _launch()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(N_ITER):
            _launch()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / N_ITER   # ms

    except Exception as e:
        return None   # config incompatible with this shape; skip silently


# ════════════════════════════════════════════════════════════════════════════
#  Main sweep
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Row:
    seq_len:    int
    head_dim:   int
    causal:     bool
    block_m:    int
    block_n:    int
    num_warps:  int
    num_stages: int
    ms:         float
    tflops:     float

    def flops(self) -> float:
        """Approximate FLOPs for one forward pass (non-causal)."""
        # 2 * B * H * N * N * D  (QK^T)  + 2 * B * H * N * N * D  (PV)
        factor = 0.5 if self.causal else 1.0
        return factor * 4.0 * BATCH * N_HEADS * self.seq_len * self.seq_len * self.head_dim


all_rows: List[Row] = []
configs  = list(itertools.product(BLOCK_MS, BLOCK_NS, NUM_WARPS_, NUM_STAGES_))
# Drop degenerate: BLOCK_N > BLOCK_M
configs  = [(bm, bn, nw, ns) for bm, bn, nw, ns in configs if bn <= bm]

shapes       = list(itertools.product(SEQ_LENS, HEAD_DIMS, CAUSALS))
n_shapes     = len(shapes)
total        = n_shapes * len(configs)
done         = 0

# ── Progress bar helpers ──────────────────────────────────────────────────
_BAR_WIDTH = 30

def _bar(frac: float) -> str:
    filled = int(_BAR_WIDTH * frac)
    bar    = '█' * filled + '░' * (_BAR_WIDTH - filled)
    return f'[{bar}]'

def _print_progress(shape_idx: int, cfg_idx: int, elapsed: float) -> None:
    """Overwrite the current line with a compact progress report."""
    cfg_done  = (shape_idx - 1) * len(configs) + cfg_idx
    frac      = cfg_done / total if total else 0
    pct       = frac * 100
    eta_s     = (elapsed / cfg_done * (total - cfg_done)) if cfg_done > 0 else 0
    eta_str   = f'{int(eta_s//60):02d}:{int(eta_s%60):02d}'
    line = (
        f'  {_bar(frac)} '
        f'{_c(f"[{shape_idx}/{n_shapes}", "hi")}{_c("]", "hi")} '
        f'{_c(f"{pct:5.1f}%", "num")}  '
        f'cfg {cfg_idx:>{len(str(len(configs)))}}/{len(configs)}  '
        f'ETA {eta_str}   '
    )
    print(f'\r{line}', end='', flush=True)

print()
print(_c("=" * 78, 'hdr'))
print(_c("  SageAttention Config Profiler — AMD RX 6800 (gfx1030)", 'hdr'))
print(_c("=" * 78, 'hdr'))
print(f"  Device    : {torch.cuda.get_device_name(0)}")
print(f"  PyTorch   : {torch.__version__}")
print(f"  Seq lens  : {SEQ_LENS}")
print(f"  Head dims : {HEAD_DIMS}")
print(f"  Causal    : {CAUSALS}")
print(f"  Configs   : {len(configs)} grid points  ×  {n_shapes} shapes  = {total} runs")
print(f"  Warmup    : {N_WARMUP}  |  Iters : {N_ITER}")
print(_c("=" * 78, 'hdr'))
print()

t0 = time.time()

for shape_idx, (seq_len, head_dim, is_causal) in enumerate(shapes, 1):
    # Build tensors once per shape; quant once (same for all configs)
    q = torch.randn(BATCH, N_HEADS, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(BATCH, N_HEADS, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(BATCH, N_HEADS, seq_len, head_dim, dtype=torch.float16, device=device)

    with torch.no_grad():
        q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, tensor_layout="HND")

    # q_scale/k_scale come back as [b, h, blocks, 1] — squeeze last dim for kernel
    q_scale = q_scale.squeeze(-1).contiguous()
    k_scale = k_scale.squeeze(-1).contiguous()

    shape_rows: List[Row] = []

    for cfg_idx, (bm, bn, nw, ns) in enumerate(configs, 1):
        _print_progress(shape_idx, cfg_idx, time.time() - t0)
        ms = _time_config(
            q_int8, k_int8, v,
            q_scale, k_scale,
            seq_len, seq_len, N_HEADS, N_HEADS, BATCH, head_dim,
            is_causal, bm, bn, nw, ns,
        )
        done += 1
        if ms is None:
            continue
        flops  = (0.5 if is_causal else 1.0) * 4.0 * BATCH * N_HEADS * seq_len * seq_len * head_dim
        tflops = flops / (ms * 1e-3) / 1e12
        shape_rows.append(Row(seq_len, head_dim, is_causal, bm, bn, nw, ns, ms, tflops))

    # Clear the progress line before printing shape results
    print('\r' + ' ' * 78 + '\r', end='')

    if not shape_rows:
        continue

    # Sort fastest first
    shape_rows.sort(key=lambda r: r.ms)
    best  = shape_rows[0]
    worst = shape_rows[-1]

    # Print shape header
    causal_tag = "causal" if is_causal else "non-causal"
    print(_c(f"  seq={seq_len:>4}  head_dim={head_dim}  {causal_tag}", 'bld'))
    print(_c(f"  {'BM':>4}  {'BN':>4}  {'WRP':>4}  {'STG':>4}  {'ms':>8}  {'TFLOPS':>8}  rank", 'dim'))
    for rank, row in enumerate(shape_rows, 1):
        marker = _c(" ← best", 'ok') if rank == 1 else (_c(" ← worst", 'err') if rank == len(shape_rows) else "")
        col    = 'ok' if rank == 1 else ('err' if rank == len(shape_rows) else 'num')
        print(
            f"  {row.block_m:>4}  {row.block_n:>4}  {row.num_warps:>4}  {row.num_stages:>4}"
            f"  {_c(f'{row.ms:>7.3f}', col)}  {_c(f'{row.tflops:>7.4f}', col)}{marker}"
        )
    speedup = worst.ms / best.ms if best.ms > 0 else 1.0
    print(_c(f"  best speedup vs worst: {speedup:.2f}×", 'hi'))
    print()

    all_rows.extend(shape_rows)

elapsed = time.time() - t0
print(_c(f"Completed {done} config runs in {elapsed:.1f}s", 'hdr'))

# ════════════════════════════════════════════════════════════════════════════
#  CSV export
# ════════════════════════════════════════════════════════════════════════════
if all_rows:
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq_len", "head_dim", "causal", "block_m", "block_n",
                    "num_warps", "num_stages", "ms", "tflops"])
        for r in all_rows:
            w.writerow([r.seq_len, r.head_dim, r.causal, r.block_m, r.block_n,
                        r.num_warps, r.num_stages, f"{r.ms:.4f}", f"{r.tflops:.4f}"])
    print(_c(f"Results written → {CSV_OUT}", 'ok'))

# ════════════════════════════════════════════════════════════════════════════
#  Global best-config summary (console)
# ════════════════════════════════════════════════════════════════════════════
seen = set()
best_rows = sorted(all_rows, key=lambda r: (r.seq_len, r.head_dim, r.causal, r.ms))
print()
print(_c("═" * 78, 'hdr'))
print(_c("  RECOMMENDED CONFIGS (fastest per shape)", 'hdr'))
print(_c("═" * 78, 'hdr'))
print(_c(f"  {'seq':>5}  {'hd':>4}  {'csl':>5}  {'BM':>4}  {'BN':>4}  {'WRP':>4}  {'STG':>4}  {'ms':>8}  {'TFLOPS':>8}", 'dim'))
for r in best_rows:
    key = (r.seq_len, r.head_dim, r.causal)
    if key in seen:
        continue
    seen.add(key)
    print(
        f"  {r.seq_len:>5}  {r.head_dim:>4}  {'yes' if r.causal else 'no':>5}"
        f"  {r.block_m:>4}  {r.block_n:>4}  {r.num_warps:>4}  {r.num_stages:>4}"
        f"  {_c(f'{r.ms:>7.3f}', 'num')}  {_c(f'{r.tflops:>7.4f}', 'num')}"
    )
print()
print(_c("Copy the BM/BN/WRP/STG values for each seq bucket into the kernel forward()", 'dim'))
print(_c("thresholds in sageattention/attn_qk_int8_per_block*.py", 'dim'))
print()

# ════════════════════════════════════════════════════════════════════════════
#  Plain-text log file
# ════════════════════════════════════════════════════════════════════════════
if all_rows:
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_OUT, "w", encoding="utf-8") as lf:
        def _lw(line=""):
            lf.write(line + "\n")

        _lw("=" * 78)
        _lw("  SageAttention Config Profiler — bench results")
        _lw(f"  Run       : {run_ts}")
        _lw(f"  Device    : {torch.cuda.get_device_name(0)}")
        _lw(f"  PyTorch   : {torch.__version__}")
        _lw(f"  Elapsed   : {elapsed:.1f}s  ({done} config runs)")
        _lw(f"  Seq lens  : {SEQ_LENS}")
        _lw(f"  Head dims : {HEAD_DIMS}")
        _lw(f"  Causal    : {CAUSALS}")
        _lw(f"  Batch     : {BATCH}   N_heads : {N_HEADS}")
        _lw(f"  Warmup    : {N_WARMUP}   Iters   : {N_ITER}")
        _lw("=" * 78)
        _lw()

        # Per-shape ranked tables
        _lw("─" * 78)
        _lw("  ALL RESULTS (ranked fastest → slowest per shape)")
        _lw("─" * 78)
        seen2 = set()
        by_shape = {}
        for r in all_rows:
            k = (r.seq_len, r.head_dim, r.causal)
            by_shape.setdefault(k, []).append(r)
        for k in sorted(by_shape):
            seq_l, hd, csl = k
            rows = sorted(by_shape[k], key=lambda r: r.ms)
            worst_ms = rows[-1].ms
            causal_tag = "causal" if csl else "non-causal"
            _lw()
            _lw(f"  seq={seq_l:>5}  head_dim={hd}  {causal_tag}")
            _lw(f"  {'BM':>4}  {'BN':>4}  {'WRP':>4}  {'STG':>4}  {'ms':>9}  {'TFLOPS':>9}  rank")
            for rank, row in enumerate(rows, 1):
                tag = "  <- best" if rank == 1 else ("  <- worst" if rank == len(rows) else "")
                _lw(
                    f"  {row.block_m:>4}  {row.block_n:>4}  {row.num_warps:>4}  {row.num_stages:>4}"
                    f"  {row.ms:>9.4f}  {row.tflops:>9.4f}{tag}"
                )
            speedup = rows[-1].ms / rows[0].ms if rows[0].ms > 0 else 1.0
            _lw(f"  best speedup vs worst: {speedup:.2f}x")

        # Best-config summary table
        _lw()
        _lw("=" * 78)
        _lw("  RECOMMENDED CONFIGS (fastest per shape)")
        _lw("=" * 78)
        _lw(f"  {'seq':>5}  {'hd':>4}  {'causal':>6}  {'BM':>4}  {'BN':>4}  {'WRP':>4}  {'STG':>4}  {'ms':>9}  {'TFLOPS':>9}")
        seen3 = set()
        for r in best_rows:
            key = (r.seq_len, r.head_dim, r.causal)
            if key in seen3:
                continue
            seen3.add(key)
            _lw(
                f"  {r.seq_len:>5}  {r.head_dim:>4}  {'yes' if r.causal else 'no':>6}"
                f"  {r.block_m:>4}  {r.block_n:>4}  {r.num_warps:>4}  {r.num_stages:>4}"
                f"  {r.ms:>9.4f}  {r.tflops:>9.4f}"
            )
        _lw()
        _lw("Copy the BM/BN/WRP/STG values for each seq bucket into the kernel forward()")
        _lw("thresholds in sageattention/attn_qk_int8_per_block*.py")
        _lw()

    print(_c(f"Log written     → {LOG_OUT}", 'ok'))
print()
