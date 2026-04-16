import torch

def _get_arch_caps():
    """Detect RDNA and TDM support once at import time."""
    try:
        if not torch.cuda.is_available():
            return False, False
        if torch.version.hip is not None:
            gcn = torch.cuda.get_device_properties(0).gcnArchName  # e.g. "gfx1030"
            is_rdna = gcn.startswith("gfx1")        # RDNA1/2/3/4 — warp_size=32, max 4 warps
            supports_tdm = gcn.startswith("gfx125") # TDM hardware only on RDNA4 (gfx1250+)
            return is_rdna, supports_tdm
        cap = torch.cuda.get_device_capability(0)   # NVIDIA: TMA from Hopper (sm90+)
        return False, cap[0] >= 9
    except Exception:
        return False, False

_is_rdna, _supports_tdm = _get_arch_caps()
