import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    gpu_name: str
    vram_gb: float
    compute_backend: str   # cuda | metal | cpu
    os: str                # linux | darwin | windows


def detect_hardware() -> HardwareInfo:
    os_name = platform.system().lower()

    # NVIDIA via nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            line = out.stdout.strip().splitlines()[0]
            name, mem_mib = line.split(",", 1)
            vram_gb = round(float(mem_mib.strip()) / 1024, 1)
            return HardwareInfo(
                gpu_name=name.strip(),
                vram_gb=vram_gb,
                compute_backend="cuda",
                os=os_name,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Apple Silicon (arm64 only — not Intel Mac)
    if os_name == "darwin" and platform.machine() == "arm64":
        try:
            chip = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            mem_bytes = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, timeout=3,
                ).stdout.strip()
            )
            # Unified memory: GPU and CPU share the same pool.
            # 75% is a conservative usable estimate for GPU workloads.
            vram_gb = round(mem_bytes / (1024 ** 3) * 0.75, 1)
            label = chip if chip else "Apple Silicon"
            return HardwareInfo(
                gpu_name=label,
                vram_gb=vram_gb,
                compute_backend="metal",
                os=os_name,
            )
        except Exception:
            pass

    return HardwareInfo(
        gpu_name="CPU only",
        vram_gb=0.0,
        compute_backend="cpu",
        os=os_name,
    )


# Known VRAM for named cloud/consumer GPUs (used when --hardware is set explicitly)
KNOWN_VRAM: dict[str, float] = {
    "t4":        16.0,
    "l4":        24.0,
    "a10g":      24.0,
    "a10":       24.0,
    "a100-40":   40.0,
    "a100-80":   80.0,
    "a100":      40.0,
    "h100":      80.0,
    "rtx4090":   24.0,
    "rtx3090":   24.0,
    "rtxa6000":  48.0,
}
