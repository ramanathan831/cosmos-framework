"""Prespecified boundary stopping for actual equal-time checkpoints.

A fixed reserve admits complete updates and delivery overhead. Overruns are
recorded as censored, never repaired by substituting a different checkpoint.
This policy neither changes optimizer steps nor infers weights from a loss curve.
"""

import math
from dataclasses import dataclass, field


@dataclass
class BudgetPolicy:
    kind: str
    seconds: float
    reserve_seconds: float
    launch_monotonic: float
    quality_target: float | None = None
    cold_offset_seconds: float = 0.0
    loop_monotonic: float | None = None
    updates: list[dict] = field(default_factory=list)
    target_hit: dict | None = None

    def __post_init__(self):
        if self.kind not in {"training", "end_to_end", "target"}:
            raise ValueError("unknown budget clock")
        if any(
            isinstance(x, bool) or not math.isfinite(x)
            for x in (self.seconds, self.reserve_seconds, self.launch_monotonic, self.cold_offset_seconds)
        ):
            raise ValueError("budget values must be finite numbers")
        if not 0 <= self.reserve_seconds < self.seconds or self.cold_offset_seconds < 0:
            raise ValueError("budget must leave a positive usable interval")
        if self.kind == "target" and (
            self.quality_target is None or not math.isfinite(self.quality_target) or self.quality_target <= 0
        ):
            raise ValueError("target runs need a positive fixed quality target")

    def elapsed(self, now):
        origin = self.loop_monotonic if self.kind == "training" else self.launch_monotonic
        if origin is None:
            return None
        return now - origin + (self.cold_offset_seconds if self.kind != "training" else 0.0)

    def update(self, step, now):
        if self.updates and step != self.updates[-1]["step"] + 1:
            raise ValueError("budget recorder observed a skipped or duplicated update")
        elapsed = self.elapsed(now)
        if elapsed is None:
            raise ValueError("training clock has not started")
        self.updates.append({"step": step, "clock_seconds": elapsed, "monotonic": now})

    def validation(self, step, nll, now):
        if not math.isfinite(nll):
            raise ValueError("nonfinite task validation")
        if self.kind == "target" and step > 0 and nll <= self.quality_target and self.target_hit is None:
            self.target_hit = {"step": step, "nll": nll, "clock_seconds": self.elapsed(now)}

    def stop_reason(self, now):
        if self.target_hit is not None:
            return "prespecified task quality target reached"
        elapsed = self.elapsed(now)
        if elapsed is not None and elapsed >= self.seconds - self.reserve_seconds:
            return "prespecified budget delivery reserve reached"
        return None

    def report(self, now):
        elapsed = self.elapsed(now)
        last = self.updates[-1] if self.updates else None
        return {
            "kind": self.kind,
            "budget_seconds": self.seconds,
            "reserve_seconds": self.reserve_seconds,
            "cold_offset_seconds": self.cold_offset_seconds,
            "updates": self.updates,
            "target_hit": self.target_hit,
            "training_checkpoint_eligible": last is not None and last["clock_seconds"] <= self.seconds,
            "process_complete_seconds": elapsed,
            "process_within_budget": elapsed is not None and elapsed <= self.seconds,
            "requires_external_delivery_check": self.kind != "training",
            "selection_rule": "Stop at the first native boundary reaching the fixed safety/delivery reserve; retain the actual native checkpoint. Censor any overrun or missing eligible update.",
        }
