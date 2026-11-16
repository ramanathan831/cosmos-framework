"""Bounded diagnostic launcher; no inherited credentials enter the profiler environment."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Explicit values needed by the already-qualified native launcher and runtime.
# Never forward arbitrary inherited variables or a prefix such as all SLURM_*.
KEYS = """PATH HOME USER LOGNAME LANG LC_ALL LD_LIBRARY_PATH PYTHONPATH
POSTTRAIN_SIM_PYTHONPATH RESULTS RESOURCE_ROOT MODEL_PYTHON SIM_PYTHON
HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE WANDB_MODE OMP_NUM_THREADS MKL_NUM_THREADS
NO_ALBUMENTATIONS_UPDATE TOKENIZERS_PARALLELISM MUJOCO_GL PYOPENGL_PLATFORM
GR00T_DIT_SDPA_MODE CUBLAS_WORKSPACE_CONFIG GPUS_PER_NODE LIBERO_MANIFEST_ROOT
LIBERO_CONFIG_PATH POSTTRAIN_SIM_PYTHON RAY_TMPDIR B1K_SIM_IPC_DIR B1K_SHM_DIR
NUMBA_CACHE_DIR XDG_CACHE_HOME CUDA_CACHE_PATH IMAGINAIRE_OUTPUT_ROOT
__EGL_VENDOR_LIBRARY_FILENAMES PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
NVIDIA_DRIVER_CAPABILITIES CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES
NCCL_DEBUG NCCL_DEBUG_FILE NCCL_DEBUG_SUBSYS NCCL_P2P_DISABLE NCCL_SHM_DISABLE
NCCL_IB_DISABLE NCCL_SOCKET_IFNAME NCCL_IB_HCA NCCL_NET NCCL_CUMEM_ENABLE
NCCL_CUMEM_HOST_ENABLE NCCL_NVLS_ENABLE NCCL_P2P_LEVEL NCCL_NET_GDR_LEVEL
NCCL_PXN_DISABLE NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF CUDA_MODULE_LOADING CUDA_DEVICE_ORDER
CUDA_DEVICE_MAX_CONNECTIONS CUDA_LAUNCH_BLOCKING TORCH_NCCL_ASYNC_ERROR_HANDLING
TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_AVOID_RECORD_STREAMS TORCH_NCCL_ENABLE_MONITORING
SLURM_JOB_ID SLURM_LOG_DIR SLURM_JOB_NODELIST SLURM_NODELIST
SLURM_NODEID SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_JOB_NUM_NODES SLURM_NNODES
SLURM_GPUS_PER_NODE SLURM_CPUS_PER_TASK SLURM_STEP_GPUS SLURM_JOB_GPUS""".split()


def profile_options(source):
    """Predeclared Nsight variants. ``um_faults`` reproduces the round-10 capture.

    ``cuda_only`` keeps CUDA API/kernel/memcpy and NVTX tracing but disables CUPTI
    unified-memory CPU/GPU fault tracing, the instrumentation suspected of blocking the
    profiled learner while the driver evicts managed pages under oversubscription.
    Any other value is rejected rather than silently profiling something else.
    """
    variant = source.get("PROFILE_VARIANT", "um_faults")
    flags = {"um_faults": "true", "cuda_only": "false"}
    if variant not in flags:
        raise ValueError(f"unknown PROFILE_VARIANT {variant!r}; expected one of {sorted(flags)}")
    return variant, [
        "--cuda-um-cpu-page-faults=" + flags[variant],
        "--cuda-um-gpu-page-faults=" + flags[variant],
    ]


def environment(source):
    return {key: source[key] for key in KEYS if key in source}


def completion(root, profiler_returncode):
    """A successful trace export is not evidence of successful native training."""
    native_returncode = None
    reason = None
    try:
        native_returncode = json.loads((root / "attempt.json").read_text())["returncode"]
        if type(native_returncode) is not int:
            raise ValueError("native returncode must be an integer")
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = f"No valid native completion receipt: {type(error).__name__}"
    passed = profiler_returncode == 0 and type(native_returncode) is int and native_returncode == 0
    (root / "profile-completion.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "profiler_returncode": profiler_returncode,
                "native_returncode": native_returncode,
                "reason": reason or (None if passed else "Profiler or native training failed"),
            },
            indent=2,
        )
        + "\n"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    root = Path(os.environ["RESULTS"])
    nsys = "/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys"
    variant, um_flags = profile_options(os.environ)
    version = subprocess.run(
        [nsys, "--version"], env=environment(os.environ), capture_output=True, text=True, check=True
    )
    (root / "profile-context.json").write_text(
        json.dumps(
            {
                "binary_sha256": hashlib.sha256(Path(nsys).read_bytes()).hexdigest(),
                "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "version": version.stdout.strip(),
                "variant": variant,
                "um_fault_flags": um_flags,
                "environment_keys": sorted(environment(os.environ)),
                "scope": "Instrumented diagnostic; native process-tree memory sampler excludes profiler ancestors; no speed claim",
            },
            indent=2,
        )
        + "\n"
    )
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        *um_flags,
        "--delay=600",
        "--duration=600",
        "--kill=none",
        "--wait=primary",
        "--trace-fork-before-exec=false",
        "--export=sqlite",
        "--output=" + str(root / "managed-memory"),
        sys.executable,
        *sys.argv[1:],
    ]
    raise SystemExit(completion(root, subprocess.call(command, env=environment(os.environ))))
