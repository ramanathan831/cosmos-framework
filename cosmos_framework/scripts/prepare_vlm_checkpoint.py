# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the packaged Cosmos3 Omni-to-Qwen3-VL converter.

The conversion implementation is native Cosmos Framework code, but model
preparation is required before either Cosmos-RL or Cosmos Framework training
can consume a public ``cosmos3_omni`` checkpoint as a VLM.  COSMOS therefore owns
one stable command in the integration package and resolves the Framework
runtime packaged by the selected backend image.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

CONVERTER_MODULE = "cosmos_framework.scripts.convert_model_to_vlm_safetensors"
PREPARATION_PYTHON_ENV = "COSMOS_FRAMEWORK_PREPARATION_PYTHON"
DEFAULT_PREPARATION_PYTHON = "/opt/venv/cosmos_framework_preparation/bin/python"
RUNTIME_PREFLIGHT_FLAG = "--cosmos-runtime-preflight"
RUNTIME_VALIDATION_MODE = "imported_converter_module"
CONVERTER_TRAINING_ENV = "COSMOS_TRAINING"


class ConverterRuntimeError(RuntimeError):
    """The image does not contain a usable, packaged converter runtime."""


@contextmanager
def _inference_only_converter_environment() -> Iterator[None]:
    """Disable optional Framework training imports during VLM conversion."""
    previous = os.environ.get(CONVERTER_TRAINING_ENV)
    os.environ[CONVERTER_TRAINING_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CONVERTER_TRAINING_ENV, None)
        else:
            os.environ[CONVERTER_TRAINING_ENV] = previous


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _module_path_in_current_interpreter() -> Path | None:
    try:
        spec = importlib.util.find_spec(CONVERTER_MODULE)
    except (ImportError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin:
        return None
    path = Path(spec.origin)
    return path.resolve() if path.is_file() else None


def _imported_module_path_in_current_interpreter() -> Path:
    """Import the converter and return its source path.

    ``find_spec`` proves only that a module file exists.  Importing it is the
    dependency-closure gate: missing or incompatible transitive requirements
    must fail the image build instead of the first model-preparation job.
    """
    module = importlib.import_module(CONVERTER_MODULE)
    module_path = Path(str(getattr(module, "__file__", ""))).resolve()
    if not module_path.is_file():
        raise ConverterRuntimeError(f"{CONVERTER_MODULE} imported without a readable module file")
    return module_path


def resolve_preparation_python(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the interpreter containing the Framework converter module."""
    if _module_path_in_current_interpreter() is not None:
        return sys.executable
    env = os.environ if environment is None else environment
    candidate = Path(env.get(PREPARATION_PYTHON_ENV, DEFAULT_PREPARATION_PYTHON)).expanduser()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ConverterRuntimeError(
            "Cosmos3 VLM preparation runtime is missing. Rebuild this backend image "
            f"with {PREPARATION_PYTHON_ENV} pointing to a packaged Python interpreter; "
            f"checked {candidate}."
        )
    # Keep the configured venv path instead of resolving its ``python``
    # symlink to the base interpreter. CPython uses the invoked path to locate
    # the adjacent pyvenv.cfg; resolving it would silently leave the packaged
    # Framework environment and make the converter unimportable.
    return str(candidate.absolute())


def converter_command(
    arguments: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Build the native converter command without mutating the environment."""
    python = resolve_preparation_python(environment)
    return [python, "-m", CONVERTER_MODULE, *arguments]


def inspect_converter_runtime(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return deterministic provenance for the packaged converter module."""
    python = resolve_preparation_python(environment)
    if Path(python).absolute() == Path(sys.executable).absolute():
        try:
            with _inference_only_converter_environment():
                module_path = _imported_module_path_in_current_interpreter()
        except (ImportError, ModuleNotFoundError) as exc:
            raise ConverterRuntimeError(f"{CONVERTER_MODULE} dependency import failed in {python}: {exc}") from exc
    else:
        probe = (
            "import importlib, json, pathlib; "
            f"m=importlib.import_module({CONVERTER_MODULE!r}); "
            "p=pathlib.Path(m.__file__).resolve(); "
            "assert p.is_file(), p; print(json.dumps({'module_path': str(p)}))"
        )
        probe_environment = dict(os.environ if environment is None else environment)
        probe_environment[CONVERTER_TRAINING_ENV] = "0"
        result = subprocess.run(
            [python, "-c", probe],
            text=True,
            capture_output=True,
            check=False,
            env=probe_environment,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise ConverterRuntimeError(
                f"{CONVERTER_MODULE} is unavailable in {python}: "
                f"{detail[-1] if detail else f'exit {result.returncode}'}"
            )
        try:
            module_path = Path(json.loads(result.stdout)["module_path"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConverterRuntimeError(f"invalid converter probe response from {python}") from exc
    return {
        "module": CONVERTER_MODULE,
        "module_path": str(module_path),
        "module_sha256": _sha256(module_path),
        "python_executable": python,
        "validation_mode": RUNTIME_VALIDATION_MODE,
    }


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments == [RUNTIME_PREFLIGHT_FLAG]:
        print(json.dumps(inspect_converter_runtime(), indent=2, sort_keys=True))
        return
    command = converter_command(arguments)
    execution_environment = dict(os.environ)
    execution_environment[CONVERTER_TRAINING_ENV] = "0"
    os.execve(command[0], command, execution_environment)


if __name__ == "__main__":
    main()
