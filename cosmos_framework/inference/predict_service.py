# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

"""HTTP adapter for a separately installed, revision-pinned Cosmos Predict 2.5.

The upstream CLI owns models, guardrails, parallelism, and its Python environment.
Requests are serialized, with one isolated CLI process group per generation.
This intentionally trades model reload latency for dependency isolation.
"""

from __future__ import annotations

import base64
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from cosmos_framework.inference.reasoner.service import resolve_media


class PredictRuntime:
    def __init__(
        self,
        root,
        revision,
        python,
        output_dir,
        *,
        model="2B/post-trained",
        checkpoint_path=None,
        multiview=False,
        context_parallel_size=1,
        offload=False,
        disable_guardrails=False,
        timeout=1800,
    ):
        self.root = Path(root).resolve()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("predict-commit must be a full Git commit")
        actual = subprocess.check_output(["git", "-C", str(self.root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(self.root), "status", "--porcelain"], text=True).strip()
        if actual != revision or dirty:
            raise ValueError("Predict source must be clean and match predict-commit")
        self.script = self.root / "examples" / ("multiview.py" if multiview else "inference.py")
        if not self.script.is_file():
            raise ValueError(f"Missing upstream entrypoint: {self.script}")
        if context_parallel_size < 1 or timeout <= 0:
            raise ValueError("context-parallel-size and timeout must be positive")
        self.python, self.model, self.revision = python, model, revision
        self.output_dir = Path(output_dir).resolve()
        self.checkpoint_path, self.multiview = checkpoint_path, multiview
        self.context_parallel_size, self.timeout = context_parallel_size, timeout
        self.offload, self.disable_guardrails = offload, disable_guardrails

    def command(self, sample_file, output_dir):
        command = [self.python]
        if self.context_parallel_size > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={self.context_parallel_size}"]
        command += [
            str(self.script),
            "-i",
            str(sample_file),
            "-o",
            str(output_dir),
            "--model",
            self.model,
            "--context-parallel-size",
            str(self.context_parallel_size),
        ]
        if self.checkpoint_path:
            command += ["--checkpoint-path", self.checkpoint_path]
        if self.offload:
            command += ["--offload-diffusion-model", "--offload-text-encoder", "--offload-tokenizer"]
        if self.disable_guardrails:
            command.append("--disable-guardrails")
        return command

    def generate(self, sample):
        request_dir = self.output_dir / uuid.uuid4().hex
        request_dir.mkdir(parents=True, exist_ok=False)
        sample_file = request_dir / "request.json"
        sample_file.write_text(json.dumps(sample) + "\n")
        with (request_dir / "generation.log").open("w") as log:
            process = subprocess.Popen(
                self.command(sample_file, request_dir / "outputs"),
                cwd=self.root,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=self.timeout)
            except BaseException:
                # Reap torchrun children as well as the launcher on timeout/cancellation.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        if return_code:
            raise RuntimeError(f"Predict failed; inspect server output {request_dir.name}/generation.log")
        videos = sorted((request_dir / "outputs").rglob("*.mp4"))
        if not videos:
            raise RuntimeError("Predict produced no video; inspect the server log for guardrail rejection")
        return videos


def prepare_sample(payload, directory, roots, *, multiview=False):
    if payload.get("stream"):
        raise ValueError("Streaming is not supported")
    sample = dict(payload.get("sample", {}))
    sample["name"] = "generation"
    for key in (
        "prompt",
        "negative_prompt",
        "inference_type",
        "num_output_frames",
        "num_steps",
        "seed",
        "guidance",
        "resolution",
        "enable_autoregressive",
        "chunk_size",
        "chunk_overlap",
    ):
        if key in payload:
            sample[key] = payload[key]
    media = payload.get("input_path", payload.get("media"))
    if "messages" in payload:
        prompts = []
        for message in payload["messages"]:
            content = message.get("content", [])
            if isinstance(content, str):
                prompts.append(content)
            else:
                for part in content:
                    if part.get("type") == "text":
                        prompts.append(part["text"])
                    elif part.get("type") in {"image_url", "video_url"}:
                        if media is not None:
                            raise ValueError("Base generation accepts one input media file")
                        media = part[part["type"]]
                        media = media.get("url") if isinstance(media, dict) else media
        sample.setdefault("prompt", "\n".join(prompts))
    if multiview:
        # Native multiview JSON has multiple paths and a different schema. It is
        # accepted only as a server-local JSON file within an explicit media root.
        if not isinstance(payload.get("sample_path"), str):
            raise ValueError("Multiview requests require sample_path to a mounted native JSON sample")
        path = Path(payload["sample_path"]).resolve()
        if not path.is_file() or not any(path.is_relative_to(Path(root).resolve()) for root in roots):
            raise ValueError("sample_path must be inside a configured media root")
        sample = json.loads(path.read_text())
        if not isinstance(sample, dict):
            raise ValueError("sample_path must contain one native sample object")
        sample["name"] = "generation"
        return sample
    # Do not accept unvalidated input paths through the nested native sample.
    media = media or sample.pop("input_path", None)
    sample.pop("prompt_path", None)
    if media is not None:
        sample["input_path"] = resolve_media(media, directory, roots)
    if not sample.get("prompt"):
        raise ValueError("prompt is required")
    sample.setdefault("inference_type", "text2world" if media is None else "image2world")
    if sample["inference_type"] not in {"text2world", "image2world", "video2world"}:
        raise ValueError("Unsupported inference_type")
    if sample["inference_type"] != "text2world" and media is None:
        raise ValueError("Conditioned generation requires input media")
    return sample


def create_app(runtime, *, allowed_roots=()):
    from fastapi import FastAPI, HTTPException

    app, lock = FastAPI(title="Cosmos Predict"), threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ready", "model_loaded": False, "execution": "isolated-per-request"}

    @app.get("/info")
    def info():
        return {"model": runtime.model, "source_commit": runtime.revision, "multiview": runtime.multiview}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": runtime.model, "object": "model", "owned_by": "cosmos"}]}

    @app.post("/infer")
    @app.post("/v1/chat/completions")
    def generate(payload: dict):
        try:
            if payload.get("model", runtime.model) != runtime.model:
                raise ValueError("Requested model does not match the configured model")
            with TemporaryDirectory(prefix="cosmos-predict-request-") as directory:
                sample = prepare_sample(payload, directory, allowed_roots, multiview=runtime.multiview)
                with lock:
                    videos = runtime.generate(sample)
                outputs = ["data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode() for path in videos]
            return {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": runtime.model,
                "videos": outputs,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": outputs[0]}, "finish_reason": "stop"}
                ],
            }
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise HTTPException(status_code=504, detail="Generation timed out") from error
        except RuntimeError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    return app
