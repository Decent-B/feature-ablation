"""`env` stage: GPU, aarch64/cu130 torch build, device capability, HF access, versions and
the hardware record (driver, CUDA, kernel, hostname, raw nvidia-smi output).

nvidia-smi is called only to record the driver/CUDA versions and its raw output; memory
telemetry never uses it (it reports "Not Supported" on the GB10)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import re
import shutil
import socket
import subprocess

from .utils import LOG, versions


def _run(cmd: list[str], timeout: int = 20) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else (r.stdout + r.stderr)
    except Exception as e:  # noqa: BLE001
        return None if isinstance(e, FileNotFoundError) else f"error: {e}"


def cuda12_packages() -> list[str]:
    """Installed distributions that pull CUDA 12 runtimes (libcudart.so.12 on a CUDA 13 box)."""
    bad = []
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if re.search(r"(^|-)cu12($|-)", name) or name.endswith("-cu12"):
            bad.append(f"{name}=={dist.version}")
    return sorted(bad)


def run_env_check(cfg: dict, model_id: str, allow_cpu: bool) -> dict:
    import torch

    rec = {"platform": {"machine": platform.machine(), "kernel": " ".join(platform.uname()),
                        "hostname": socket.gethostname(), "python": platform.python_version()},
           "versions": versions(), "warnings": [], "gpu_only_checks_untested": False}
    smi = _run(["nvidia-smi"]) if shutil.which("nvidia-smi") else None
    rec["nvidia_smi_raw"] = smi
    if smi:
        m = re.search(r"Driver Version:\s*([\d.]+)", smi)
        c = re.search(r"CUDA Version:\s*([\d.]+)", smi)
        rec["driver_version"] = m.group(1) if m else None
        rec["driver_cuda_version"] = c.group(1) if c else None
    rec["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "cu130_build": "+cu130" in torch.__version__ or (torch.version.cuda or "").startswith("13")}
    bad = cuda12_packages()
    rec["cuda12_packages"] = bad
    if bad:
        msg = (f"CUDA 12 wheels installed: {bad}. Only CUDA 13 runtimes exist on the GB10, so these fail "
               "with libcudart.so.12 import errors. Remove them and reinstall from the cu130 index.")
        if torch.cuda.is_available():
            raise SystemExit("ERROR: " + msg)
        rec["warnings"].append(msg)
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        rec["device"] = {"name": torch.cuda.get_device_name(0), "capability": list(cap)}
        if tuple(cap) != (12, 1):
            rec["warnings"].append(f"device capability {cap}, expected (12, 1) on the GB10")
        if tuple(cap) in ((12, 0), (12, 1)):
            LOG.info("note: PyTorch may warn that capability %s is outside its supported range; sm_120 and "
                     "sm_121 are binary compatible, so the warning is harmless (logged once).", cap)
        a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        b = (a @ a.T).float()
        torch.cuda.synchronize()
        rec["bf16_matmul_ok"] = bool(torch.isfinite(b).all())
        if not rec["torch"]["cu130_build"]:
            rec["warnings"].append(f"torch {torch.__version__} is not a cu130 build")
        LOG.info("GPU %s capability %s, torch %s (CUDA %s), bf16 matmul %s", rec["device"]["name"], cap,
                 torch.__version__, torch.version.cuda, "ok" if rec["bf16_matmul_ok"] else "FAILED")
    else:
        if not allow_cpu:
            raise SystemExit("ERROR: torch.cuda.is_available() is False. A libcudart.so.12 error means a CUDA 12 "
                             "wheel slipped in; if the cu130 wheels fail, use NVIDIA's NGC PyTorch container.")
        rec["device"] = {"name": "cpu", "capability": None}
        rec["gpu_only_checks_untested"] = True
        rec["warnings"].append("no GPU: CPU smoke path. cu130 build, capability (12, 1), bf16 matmul, Triton, "
                               "and GPU memory telemetry are untested on this machine.")
        LOG.warning("no GPU visible: GPU-only checks (cu130 build, capability (12,1), bf16 matmul) are untested")
    ptxas = os.environ.get("TRITON_PTXAS_PATH")
    rec["triton_ptxas_path"] = {"value": ptxas, "exists": bool(ptxas and os.path.exists(ptxas))}
    rec["systemd_scope"] = {"active": os.environ.get("PILOT_SYSTEMD_SCOPE") == "1",
                            "note": os.environ.get("PILOT_SYSTEMD_NOTE", "")}
    # optional fused kernels for the hybrid Qwen3.5 model (actual implementation logged at load time)
    for pkg in ("fla", "causal_conv1d", "flash_attn"):
        try:
            __import__(pkg)
            rec.setdefault("optional_kernels", {})[pkg] = "importable"
        except Exception as e:  # noqa: BLE001
            rec.setdefault("optional_kernels", {})[pkg] = f"not importable ({type(e).__name__})"
    if rec["optional_kernels"].get("flash_attn") == "importable":
        rec["warnings"].append("flash-attn is installed but unused: attention is SDPA only")
    tok_present = bool(os.environ.get("HF_TOKEN")) or os.path.exists(
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token"))
    rec["hf"] = {"token_present": tok_present, "offline": os.environ.get("HF_HUB_OFFLINE") == "1"}
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id)
        rec["hf"]["model_access"] = True
        rec["hf"]["model_sha"] = info.sha
        rec["hf"]["gated"] = bool(getattr(info, "gated", False))
    except Exception as e:  # noqa: BLE001
        rec["hf"]["model_access"] = False
        rec["hf"]["error"] = repr(e)[:300]
        rec["warnings"].append(f"could not reach the Hugging Face API for {model_id}: {e!r}"[:300])
    for w in rec["warnings"]:
        LOG.warning("env: %s", w)
    return rec
