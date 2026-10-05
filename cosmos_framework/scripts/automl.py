# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Framework AutoML CLI; GPU dependencies are loaded only by trial commands."""


def main() -> None:
    try:
        from automl_core.cli import main as core_main
    except ModuleNotFoundError as exc:
        if exc.name not in {"automl_core", "automl_core.cli"}:
            raise
        raise SystemExit("Install the optional AutoML dependencies: pip install 'cosmos-framework[automl]'") from exc
    from cosmos_framework.automl.workload import load

    core_main(loader=load)


if __name__ == "__main__":
    main()
