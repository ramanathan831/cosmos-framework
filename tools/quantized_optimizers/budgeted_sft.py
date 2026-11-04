"""Native SFT terminated at a declared elapsed budget or task-quality target.

The regular termination/checkpoint transaction owns saving. Final scoring must
load the actual committed checkpoint, and its cost is added by the caller before
an end-to-end-budget result can pass. Numeric budgets are locked in a separate
protocol before any final scoring; this module does not choose them.
"""

import json
import os
import time
from pathlib import Path

from tools.quantized_optimizers.budget_policy import BudgetPolicy


def main():
    spec_path = Path(os.environ["OPTIMIZER_BUDGET_SPEC"])
    spec = json.loads(spec_path.read_text())
    if "PRODUCT_LAUNCH_MONOTONIC" not in os.environ:
        raise ValueError("caller must provide the clock before input staging/loading")
    policy = BudgetPolicy(**spec, launch_monotonic=float(os.environ["PRODUCT_LAUNCH_MONOTONIC"]))
    from cosmos_framework.posttrain.control.termination import TerminationWatch
    from tools.quantized_optimizers import task_development as task

    task.install()
    entry = task.entry
    original_loop, original_step, original_evaluate = entry.loop, entry.train_step, entry._evaluate
    original_reason = TerminationWatch._local_reason

    def loop(*args, **kwargs):
        policy.loop_monotonic = time.monotonic()
        return original_loop(*args, **kwargs)

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        policy.update(int(result["optim/optimizer_step"]), time.monotonic())
        return result

    def evaluate(*args, **kwargs):
        report = original_evaluate(*args, **kwargs)
        policy.validation(report.model_step, report.metrics["val/task_answer_nll"], time.monotonic())
        return report

    def reason(watch):
        return original_reason(watch) or policy.stop_reason(time.monotonic())

    entry.loop, entry.train_step, entry._evaluate = loop, step, evaluate
    TerminationWatch._local_reason = reason
    entry.main()
    report = policy.report(time.monotonic())
    checkpoints = list(Path(os.environ["IMAGINAIRE_OUTPUT_ROOT"]).rglob("model.pt"))
    report["committed_model_paths"] = [str(path.parent) for path in checkpoints]
    report["spec"] = spec
    if len(checkpoints) != 1:
        raise ValueError(f"budget run must produce one actual native model checkpoint, found {checkpoints}")
    (task.root / "budget.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
