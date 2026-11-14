"""Read-only action runtime import and dependency inventory (no task qualification)."""

import argparse
import importlib
import importlib.metadata
import json
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = {}
    for name in (
        "gr00t.model.gr00t_n1d7.gr00t_n1d7",
        "vllm",
        "vllm_omni",
        "bitsandbytes",
        "cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline",
        "cosmos_framework.posttrain.entrypoints.joint_ppo",
    ):
        try:
            module = importlib.import_module(name)
            results[name] = {"ok": True, "file": module.__file__}
        except Exception:
            results[name] = {"ok": False, "traceback": traceback.format_exc()}
        print(json.dumps({name: results[name]["ok"]}), flush=True)
        args.output.write_text(json.dumps({"imports": results, "complete": False}, indent=2))
    versions = {
        name: importlib.metadata.version(name)
        for name in {d.metadata["Name"] for d in importlib.metadata.distributions()}
    }
    args.output.write_text(json.dumps({"imports": results, "versions": versions}, indent=2))
    print(json.dumps({k: v["ok"] for k, v in results.items()}))
    if not all(v["ok"] for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
