"""Read-only action runtime import and dependency inventory (no task qualification)."""

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path


def probe():
    results = {}
    for name in (
        "gr00t.model.gr00t_n1d7.gr00t_n1d7",
        "vllm",
        "vllm_omni",
        "vllm_omni.diffusion.distributed.sp_plan",
        "bitsandbytes",
        "cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline",
        "cosmos_framework.posttrain.entrypoints.joint_ppo",
    ):
        print(json.dumps({"importing": name}), flush=True)
        try:
            module = importlib.import_module(name)
            results[name] = {"ok": True, "file": module.__file__}
        except Exception:
            results[name] = {"ok": False, "traceback": traceback.format_exc()}
        print(json.dumps({name: results[name]["ok"]}), flush=True)
    versions = {
        name: importlib.metadata.version(name)
        for name in {d.metadata["Name"] for d in importlib.metadata.distributions()}
    }
    return {"imports": results, "versions": versions, "python": sys.executable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ray-worker", action="store_true", help="Also check the native actor's actual import path")
    args = parser.parse_args()
    result = probe()
    if args.ray_worker:
        import ray

        from cosmos_framework.posttrain.workers.grid import _actor_pythonpath
        from cosmos_framework.posttrain.workers.ray import _Host

        paths = _actor_pythonpath(None)
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
        try:
            host = _Host.options(runtime_env={"env_vars": {"PYTHONPATH": os.pathsep.join(paths)}}).remote(paths)
            result["native_actor"] = ray.get(host.run.remote(lambda _: probe()))
            # Ray adds its vendored thirdparty_files (e.g. colorama) to workers.
            # Record extras, but require every driver dependency and imported
            # module to retain the exact identity; extras cannot replace a pin.
            result["actor_extra_distributions"] = sorted(
                set(result["native_actor"]["versions"]) - set(result["versions"])
            )
        finally:
            ray.shutdown()
    args.output.write_text(json.dumps(result, indent=2))
    results = result["imports"]
    print(json.dumps({k: v["ok"] for k, v in results.items()}))
    if not all(v["ok"] for v in results.values()):
        raise SystemExit(1)
    if args.ray_worker:
        worker = result["native_actor"]
        if worker["imports"] != result["imports"] or any(
            worker["versions"].get(name) != version for name, version in result["versions"].items()
        ):
            raise RuntimeError("native actor dependency/module identity differs from the driver")


if __name__ == "__main__":
    main()
