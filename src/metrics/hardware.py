import platform
import subprocess
from typing import Dict

import torch


def get_git_commit_hash() -> str:
    """
    Return the current git commit hash.

    If the code is not inside a git repository, return 'unknown' instead of
    crashing. Benchmark code should still run, but the missing commit hash
    should be visible in the raw CSV.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_cuda_version() -> str:
    """
    Return the CUDA version reported by PyTorch, or 'none' on CPU-only systems.
    """
    if torch.version.cuda is None:
        return "none"
    return str(torch.version.cuda)


def get_gpu_name() -> str:
    """
    Return the first CUDA GPU name if available, otherwise 'cpu'.
    """
    if not torch.cuda.is_available():
        return "cpu"

    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown_cuda_device"


def get_driver_version() -> str:
    """
    Return NVIDIA driver version if nvidia-smi is available.

    On CPU-only machines or systems without nvidia-smi, return 'none'.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.strip().splitlines()[0]
        return first_line.strip()
    except Exception:
        return "none"


def get_hardware_info() -> Dict[str, str]:
    """
    Return hardware and software metadata for benchmark logging.
    """
    return {
        "gpu_name": get_gpu_name(),
        "cuda_version": get_cuda_version(),
        "pytorch_version": torch.__version__,
        "driver_version": get_driver_version(),
        "commit_hash": get_git_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def raw_csv_hardware_fields() -> Dict[str, str]:
    """
    Return only the fields required by the raw CSV schema.
    """
    info = get_hardware_info()
    return {
        "gpu_name": info["gpu_name"],
        "cuda_version": info["cuda_version"],
        "pytorch_version": info["pytorch_version"],
        "driver_version": info["driver_version"],
        "commit_hash": info["commit_hash"],
    }
