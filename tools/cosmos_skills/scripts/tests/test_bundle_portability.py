# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Exercise discovery and action resolution without the former skill repository."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK = ROOT.parents[1]


@pytest.fixture(scope="module")
def relocated(tmp_path_factory):
    destination = tmp_path_factory.mktemp("relocated skills") / "bundle with spaces"
    shutil.copytree(
        ROOT,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"),
    )
    return destination


def run_helper(bundle: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment.pop("COSMOS_SKILLS_ROOT", None)
    environment.pop("PYTHONPATH", None)
    # A stale globally installed plugin must not take ownership of this bundle.
    environment["TAO_SKILL_BANK_PATH"] = "/nonexistent/deprecated-skill-bank"
    return subprocess.run(
        [sys.executable, str(bundle / script), *args],
        cwd=bundle.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


@pytest.mark.parametrize(
    ("model", "backend", "expected_backend"),
    [
        ("nvidia/Cosmos3-Nano", "cosmos-framework", "cosmos-framework"),
        ("nvidia/Cosmos3-Nano", "auto", "cosmos-rl"),
        ("nvidia/Cosmos3-Edge", "auto", "cosmos-framework"),
    ],
)
def test_relocated_resolver_owns_model_contracts(relocated, model, backend, expected_backend):
    output = run_helper(
        relocated,
        "scripts/resolve_tao_model.py",
        "--model",
        model,
        "--action",
        "train",
        "--backend",
        backend,
        "--workload",
        "training",
        "--format",
        "json",
    )
    result = json.loads(output.stdout)
    assert result["selected_backend"] == expected_backend
    assert str(relocated) in output.stdout
    assert "deprecated-skill-bank" not in output.stdout
    assert "cosmos3-reasoner" in output.stdout


def test_relocated_catalog_contains_only_bundled_models_and_all_platforms(relocated):
    result = json.loads(run_helper(relocated, "scripts/list_tao_capabilities.py", "--format", "json").stdout)
    assert {model["model"] for model in result["model_workflows"]["models"]} == {
        "cosmos3-reasoner",
        "cosmos-embed",
    }
    assert {platform["name"] for platform in result["platforms"]["supported"]} == {
        "docker",
        "slurm",
        "kubernetes",
        "brev",
        "virtualenv",
    }
    for group in ("applications", "data_workflows"):
        for item in result[group]:
            assert (relocated / item["path"]).is_file()


@pytest.mark.parametrize(
    "helper",
    [
        "scripts/resolve_tao_image.py",
        "scripts/check_tao_launch_preflight.py",
        "skills/models/cosmos3-reasoner/scripts/cosmos_workflow.py",
        "skills/models/cosmos3-reasoner/scripts/evaluation_workflow.py",
        "skills/models/cosmos3-reasoner/scripts/framework_checkpoint_action.py",
        "skills/applications/cosmos-deft-aoi/scripts/submit_cfw_train.py",
        "skills/applications/cosmos-deft-aoi/scripts/submit_cfw_evaluate.py",
        "skills/applications/cosmos-deft-traffic/scripts/prepare_cosmos_embed_inference.py",
        "skills/data/cosmos-finetune-anomalygennext/scripts/prepare_finetune_recipe.py",
        "skills/data/cosmos-prepare-anomalygennext-inputs/scripts/prepare_anomalygennext_inputs.py",
        "skills/data/cosmos-prepare-anomalygennext-inputs/scripts/run_anomalygennext_amp.py",
        "skills/data/cosmos-generate-od-defects/scripts/generate_od_defects.py",
        "skills/data/cosmos-generate-image-embeddings/scripts/verify_image_embeddings_spec.py",
    ],
)
def test_entrypoints_import_without_framework_or_skill_bank_checkout(relocated, helper):
    result = run_helper(relocated, helper, "--help")
    assert "usage:" in result.stdout.lower()


def test_env_setup_is_relocatable_and_does_not_change_working_directory(relocated):
    environment = os.environ.copy()
    environment.pop("COSMOS_SKILLS_ROOT", None)
    result = subprocess.run(
        ["bash", "-c", 'source "$1/env.sh"; printf "%s\\n%s\\n" "$COSMOS_SKILLS_ROOT" "$PWD"', "bash", str(relocated)],
        cwd=relocated.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == [str(relocated), str(relocated.parent)]


def test_agent_discovery_links_resolve_to_canonical_packages():
    skills = sorted((ROOT / "skills").glob("*/*/SKILL.md"))
    for skill in skills:
        frontmatter = yaml.safe_load(skill.read_text().split("---", 2)[1])
        assert frontmatter["name"] == skill.parent.name
        for discovery in (".agents", ".claude"):
            link = FRAMEWORK / discovery / "skills" / skill.parent.name
            assert link.is_symlink(), link
            assert not Path(os.readlink(link)).is_absolute(), link
            assert link.resolve() == skill.parent


def test_provenance_records_only_packaged_files():
    manifest = json.loads((ROOT / "migration.json").read_text())
    assert len(manifest["source_commit"]) == 40
    destinations = [entry["destination"] for entry in manifest["files"]]
    assert len(destinations) == len(set(destinations))
    for entry in manifest["files"]:
        assert (ROOT / entry["destination"]).is_file(), entry
        assert len(entry["source_sha256"]) == 64


def test_local_markdown_references_resolve_without_the_source_repository():
    for document in ROOT.rglob("*.md"):
        if any(part.startswith(".") for part in document.relative_to(ROOT).parts):
            continue
        for match in re.finditer(r"\]\(([^)]+)\)", document.read_text()):
            target = match.group(1).split("#", 1)[0]
            if not target or "://" in target or target.startswith(("/", "$", "<")):
                continue
            assert (document.parent / target).exists(), (document, target)
