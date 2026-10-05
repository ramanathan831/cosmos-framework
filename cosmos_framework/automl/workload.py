# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Reasoner SFT → native DCP checkpoint → held-out reasoner evaluation."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import tomli_w
from automl_core.contracts import Contract, Execution, Result, Search, TrialPlan, lookup
from automl_core.runner import Experiment
from automl_core.workload import local_artifact, read_manifest
from pydantic import Field

from cosmos_framework.automl.evaluate_trial import read_toml, validate_dcp
from cosmos_framework.configs.toml_config.sft_config import SFTExperimentConfig

# These knobs are consumed by both Framework's TOML schema and its VLM remapping.
# Fidelity, topology, data, identity and checkpoint lineage are not search axes.
SEARCHABLE = {
    "optimizer.lr",
    "optimizer.weight_decay",
    "trainer.grad_accum_iter",
    "dataloader_train.max_samples_per_batch",
}


class SFTWorkload:
    def __init__(self, cwd: Path, evaluation: dict, nproc: int, seed: int, experiment_id: str):
        self.cwd = cwd
        self.evaluation = evaluation
        self.nproc = nproc
        self.seed = seed
        self.experiment_id = experiment_id

    def identity(self) -> dict:
        return {
            "adapter": "cosmos-reasoner-sft-v1",
            "cwd": str(self.cwd),
            "evaluation": self.evaluation,
            "nproc": self.nproc,
            "training_seed": self.seed,
        }

    def validate(self, config: dict) -> None:
        parsed = SFTExperimentConfig.model_validate(config)
        if parsed.job.task != "vlm" or not parsed.job.experiment:
            raise ValueError("Reasoner AutoML needs a VLM SFT recipe with an explicit job.experiment")
        if "lora_enabled" not in config.get("model", {}):
            raise ValueError("Set model.lora_enabled explicitly so checkpoint evaluation matches training")
        if parsed.trainer.num_epochs is not None:
            raise ValueError("This adapter compares a fixed trainer.max_iter budget; remove num_epochs")
        maximum = config.get("trainer", {}).get("max_iter", 0)
        interval = config.get("checkpoint", {}).get("save_iter", 0)
        if maximum <= 0 or interval <= 0 or maximum % interval:
            raise ValueError("Set positive max_iter and save_iter with a checkpoint at the final iteration")
        if parsed.checkpoint.save_freq_in_epoch:
            raise ValueError("Use iteration-based checkpointing for a fixed max_iter search")
        if "load_path" not in config.get("checkpoint", {}) or "???" in json.dumps(config):
            raise ValueError("Resolve the initial checkpoint and all recipe placeholders before planning")
        # Native VLM initialization loads HF backbone weights when load_path is
        # explicitly empty; a nonempty value instead selects a DCP warm start.
        # Do not force a safetensors directory through the DCP loader.
        if not parsed.checkpoint.load_path and not config.get("model", {}).get("backbone", {}).get("model_name"):
            raise ValueError(
                "HF initialization needs an explicit model.backbone.model_name and checkpoint.load_path=''"
            )
        if parsed.optimizer.lr <= 0 or parsed.optimizer.weight_decay < 0:
            raise ValueError("Learning rate must be positive; weight decay must be non-negative")
        batch_size = parsed.dataloader_train.max_samples_per_batch
        if parsed.trainer.grad_accum_iter < 1 or (batch_size is not None and batch_size < 1):
            raise ValueError("Batch size and gradient accumulation must be positive")
        if self.evaluation.get("task", {}).get("type") not in {
            "its_directionality",
            "metropolis_sgd",
            "mcq",
            "bcq",
            "binary",
        }:
            raise ValueError("Select a scored reasoner task: mcq, bcq, binary, its_directionality or metropolis_sgd")
        if not self.evaluation.get("dataset", {}).get("annotation_path"):
            raise ValueError("An explicit held-out evaluation dataset is required")
        if parsed.model.lora_enabled and not self.evaluation.get("model", {}).get("base_model_path"):
            raise ValueError("LoRA evaluation needs model.base_model_path for native checkpoint preparation")

    def plan(self, config: dict, trial: Path, execution: Execution) -> TrialPlan:
        recipe = copy.deepcopy(config)
        recipe.setdefault("job", {}).update(project="automl", group=self.experiment_id, name=trial.name)
        recipe["job"].setdefault("wandb_mode", "disabled")
        output = trial / "train"
        run_dir = output / "automl" / self.experiment_id / trial.name
        evaluation = copy.deepcopy(self.evaluation)
        evaluation.setdefault("model", {})["enable_lora"] = recipe.get("model", {}).get("lora_enabled", False)
        command = [execution.python, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={self.nproc}"]
        command += ["-m", "cosmos_framework.scripts.train", f"--sft-toml={trial / 'config.toml'}"]
        command += [
            "--",
            f"trainer.seed={self.seed}",
            "trainer.num_epochs=null",
            "checkpoint.save_freq_in_epoch=0",
            "checkpoint.load_training_state=false",
        ]
        return TrialPlan(
            commands=[
                command,
                [
                    execution.python,
                    "-m",
                    "cosmos_framework.automl.evaluate_trial",
                    "--run-dir",
                    str(run_dir),
                    "--evaluation-config",
                    str(trial / "evaluation.toml"),
                    "--output-dir",
                    str(trial / "evaluation"),
                    "--expected-iteration",
                    str(recipe["trainer"]["max_iter"]),
                ],
            ],
            cwd=self.cwd,
            files={"config.toml": tomli_w.dumps(recipe), "evaluation.toml": tomli_w.dumps(evaluation)},
            environment={"IMAGINAIRE_OUTPUT_ROOT": str(output), "PYTHONHASHSEED": str(self.seed)},
        )

    def collect(self, trial: Path, objective) -> Result:
        metrics_path = local_artifact(trial, "evaluation/metrics/metrics.json")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("overall", {}).get("total", 0) <= 0:
            raise ValueError("An empty evaluation is not an observation")
        value = lookup(metrics, objective.metric)
        if type(value) not in (float, int):
            raise ValueError("The objective must be a finite JSON number")
        handoff = json.loads(local_artifact(trial, "evaluation/checkpoint.json").read_text())
        checkpoint = local_artifact(trial, handoff["checkpoint"])
        validate_dcp(checkpoint)
        return Result(
            metric=value,
            artifacts={
                "checkpoint": str(checkpoint),
                "config": str(local_artifact(trial, "config.toml")),
                "metrics": str(metrics_path),
            },
        )


class WorkloadSettings(Contract):
    recipe: Path
    evaluation: Path
    cwd: Path
    nproc_per_node: int = Field(ge=1, strict=True)


class Manifest(Contract):
    search: Search
    workload: WorkloadSettings
    execution: Execution


def load(path: Path, workspace: Path) -> Experiment:
    raw = read_manifest(path)
    if "base_config" in raw.get("search", {}):
        raise ValueError("Use workload.recipe as the single source of the Framework base configuration")
    recipe_path = (path.parent / raw["workload"]["recipe"]).resolve()
    raw["search"]["base_config"] = read_toml(recipe_path)
    manifest = Manifest.model_validate(raw)
    unsupported = set(manifest.search.parameters) - SEARCHABLE
    if unsupported:
        raise ValueError(f"Search axes need adapter support: {sorted(unsupported)}; supported: {sorted(SEARCHABLE)}")
    evaluation = read_toml((path.parent / manifest.workload.evaluation).resolve())
    # Hydra reads these only inside the selected execution environment; persist names, not values.
    required = set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)[^}]*\}", json.dumps(raw)))
    if not required.issubset(manifest.execution.environment):
        raise ValueError(f"Declare recipe environment names in execution.environment: {sorted(required)}")
    manifest.execution.mounts = [(path.parent / p).resolve() for p in manifest.execution.mounts]
    workload = SFTWorkload(
        (path.parent / manifest.workload.cwd).resolve(),
        evaluation,
        manifest.workload.nproc_per_node,
        manifest.search.seed,
        manifest.search.experiment_id,
    )
    return Experiment(manifest.search, workload, manifest.execution, workspace)
