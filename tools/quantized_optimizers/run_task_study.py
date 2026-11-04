"""Run an explicitly reserved development/evaluation command list, keeping raw costs."""

import json
import os
import subprocess
import threading
import time

from tools.quantized_optimizers.run_deployment import monitor, read_model_files, write


def run(root, staging, handle, deadline):
    spec = json.loads(open(os.environ["TASK_SPEC"]).read())
    if spec["purpose"] != "development diagnosis":
        raise ValueError("this launcher is currently restricted to development diagnosis")
    mapping = {"data": os.environ["SFT_DATA_ROOT"], "prepared": str(staging), "root": str(root)}
    for case in spec["cases"]:
        output = root / case["name"]
        output.mkdir(exist_ok=False)
        mapping["case"] = str(output)
        argv = [value.format_map(mapping) for value in case["argv"]]
        env = dict(os.environ, SFT_PREPARED_CORPUS=str(staging))
        env.update({key: value.format_map(mapping) for key, value in case.get("env", {}).items()})
        write(output / "command.json", {"argv": argv, "overrides": case.get("env", {}), "purpose": spec["purpose"]})
        before = time.monotonic()
        if case.get("read_model", True):
            read_model_files(output)
        with (output / "process.log").open("x") as log:
            process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
            stop = threading.Event()
            thread = threading.Thread(target=monitor, args=(handle, process, output / "memory.jsonl", stop))
            thread.start()
            try:
                code = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=30)
                raise
            finally:
                stop.set()
                thread.join()
        write(output / "attempt.json", {"returncode": code, "elapsed_seconds": time.monotonic() - before})
        if code and not case.get("allow_failure", False):
            raise RuntimeError(f"development case failed: {case['name']}")
    write(root / "complete.json", {"cases": [case["name"] for case in spec["cases"]]})
