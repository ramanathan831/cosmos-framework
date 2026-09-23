# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serve an explicitly installed, pinned native Cosmos Predict 2.5 checkout."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict-root", required=True)
    parser.add_argument("--predict-commit", required=True)
    parser.add_argument("--predict-python", default=sys.executable)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="2B/post-trained")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--multiview", action="store_true")
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--disable-guardrails", action="store_true")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--media-root", action="append", default=[])
    args = parser.parse_args()

    import uvicorn

    from cosmos_framework.inference.predict_service import PredictRuntime, create_app

    runtime = PredictRuntime(
        args.predict_root,
        args.predict_commit,
        args.predict_python,
        args.output_dir,
        model=args.model,
        checkpoint_path=args.checkpoint_path,
        multiview=args.multiview,
        context_parallel_size=args.context_parallel_size,
        offload=args.offload,
        disable_guardrails=args.disable_guardrails,
        timeout=args.timeout,
    )
    uvicorn.run(create_app(runtime, allowed_roots=args.media_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
