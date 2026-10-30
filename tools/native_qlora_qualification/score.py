# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded WTS evaluation through sealed native Cosmos APIs; never trains.

Run separately for each frozen or adapter-loaded arm under torchrun world-size 1.
The default selects at most 256 rows with seed 2026 before constructing the model.
This is teacher-forced next-token scoring, not free-generation task accuracy.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

PROTOCOL = "cosmos.wts-heldout.v1"
LETTERS = ("A", "B", "C", "D")
OPTION = re.compile(r"(?m)^[ \t]*([A-Z])([:.])[ \t]+(?=\S)")
OPTION_MARKER = re.compile(r"(?m)^[ \t]*([A-Za-z])([:.])")


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def validate_selection_receipt(args: argparse.Namespace, selection: dict[str, Any]) -> None:
    """Bind final/untouched scoring to a selection fixed before its outcomes."""
    if selection.get("split_role") not in {"final", "untouched_diagnostic"}:
        return
    path, expected = getattr(args, "selection_receipt", None), getattr(args, "selection_receipt_sha256", None)
    if not path or not expected or file_digest(Path(path)) != expected:
        raise ValueError("untouched scoring requires the exact precommitted selection receipt and SHA256")
    receipt = json.loads(Path(path).read_text())
    keys = {
        "protocol",
        "split_manifest_sha256",
        "development_manifest_sha256",
        "locked_utc",
        "selected_scoring_config_sha256",
        "development_evidence_sha256",
    }
    if set(receipt) != keys or receipt["protocol"] != "cosmos.qlora-selection.v1":
        raise ValueError("unsupported selection receipt")
    if receipt["split_manifest_sha256"] != selection["split_manifest_sha256"]:
        raise ValueError("selection receipt names a different untouched split")
    stamp = datetime.datetime.fromisoformat(receipt["locked_utc"])
    if stamp.tzinfo is None or stamp > datetime.datetime.now(datetime.timezone.utc):
        raise ValueError("selection lock must have a timezone and precede scoring")
    hashes = [receipt["split_manifest_sha256"], receipt["development_manifest_sha256"]]
    for key in ("selected_scoring_config_sha256", "development_evidence_sha256"):
        if not isinstance(receipt[key], list) or not receipt[key]:
            raise ValueError("selection requires configuration and development-evidence identities")
        hashes.extend(receipt[key])
    if any(not isinstance(value, str) or re.fullmatch("[0-9a-f]{64}", value) is None for value in hashes):
        raise ValueError("selection receipt contains an invalid digest")
    if receipt["development_manifest_sha256"] == receipt["split_manifest_sha256"]:
        raise ValueError("development and untouched split must differ")
    configuration = json_digest({"config": args.config, "overrides": args.override})
    if configuration not in receipt["selected_scoring_config_sha256"]:
        raise ValueError("scoring configuration was not selected using development evidence")
    selection["selection_receipt_sha256"] = expected


def select_scoring_rows(args: argparse.Namespace, source: Path):
    if getattr(args, "all_rows", False):
        from tools.native_qlora_qualification.split import select_sealed_rows

        if not all(getattr(args, key, None) for key in ("split_manifest", "split_manifest_sha256", "split")):
            raise ValueError("all-rows scoring requires an explicit split manifest, digest and role")
        rows, selection = select_sealed_rows(source, Path(args.split_manifest), args.split_manifest_sha256)
        if selection["split_role"] != args.split:
            raise ValueError("requested split role differs from sealed manifest")
        for _, row in rows:
            validate_wts(row)
        validate_selection_receipt(args, selection)
        return rows, selection
    if any(
        getattr(args, key, None)
        for key in ("split_manifest", "split_manifest_sha256", "split", "selection_receipt", "selection_receipt_sha256")
    ):
        raise ValueError("sealed split/selection arguments require all-rows scoring")
    return select_rows(source, args.max_rows, args.seed)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_choices(choices: tuple[str, ...]) -> None:
    if not 2 <= len(choices) <= len(LETTERS) or choices != LETTERS[: len(choices)]:
        raise ValueError(
            "WTS requires two to four ordered contiguous choices starting at A; no gaps, extra, or duplicate letters"
        )


def parse_options(question: str) -> tuple[str, ...]:
    # Inspect every marker, including empty/malformed options. Otherwise an empty
    # final D option could silently turn an invalid four-choice row into A/B/C.
    matches = list(OPTION_MARKER.finditer(question))
    choices = tuple(match[1] for match in matches)
    validate_choices(choices)
    if any(OPTION.match(question, match.start()) is None for match in matches):
        raise ValueError("each WTS option marker requires whitespace and nonempty text on the same line")
    if len({match[2] for match in matches}) != 1:
        raise ValueError("WTS option lines must consistently use ':' or '.'")
    if not question[: matches[0].start()].strip():
        raise ValueError("WTS question before the options must not be empty")
    return choices


def validate_wts(row: Any) -> tuple[str, str, tuple[str, ...], str]:
    if not isinstance(row, dict) or isinstance(row.get("id"), bool) or not isinstance(row.get("id"), (str, int)):
        raise TypeError("each WTS row requires a string/integer id")
    identifier = str(row["id"])
    if not identifier:
        raise ValueError("WTS row id must not be empty")
    messages = row.get("messages")
    if not isinstance(messages, list) or any(not isinstance(message, dict) for message in messages):
        raise ValueError("WTS requires a messages list")
    roles = [message.get("role") for message in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError("WTS scorer supports one user/assistant exchange and an optional leading system message")
    if any(message.get("tool_calls") or message.get("reasoning_content") for message in messages) or row.get("tools"):
        raise ValueError("WTS scorer does not accept tool calls or reasoning-content answers")
    answer = messages[-1].get("content")
    if not isinstance(answer, str) or answer not in LETTERS:
        raise ValueError("WTS gold assistant answer must be exactly one uppercase letter A/B/C/D")
    content = messages[-2].get("content")
    if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
        raise ValueError("WTS user content requires native structured text and video")
    videos = [part.get("video") for part in content if part.get("type") == "video"]
    if len(videos) != 1 or not isinstance(videos[0], str) or not videos[0]:
        raise ValueError("each WTS question must condition on exactly one local video")
    if any(part.get("type") not in {"text", "video"} for part in content):
        raise ValueError("WTS scorer supports only text and video content parts")
    texts = [part.get("text") for part in content if part.get("type") == "text"]
    if not texts or any(not isinstance(text, str) for text in texts):
        raise ValueError("WTS question requires nonempty text content")
    choices = parse_options("\n".join(texts))
    if answer not in choices:
        raise ValueError("WTS gold answer must belong to the actual question choices")
    if row.get("opener", "self") != "self":
        raise ValueError("single-letter WTS answers require opener=self")
    return identifier, answer, choices, videos[0]


def select_rows(source: Path, max_rows: int, seed: int) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    """Bound retained row payloads by max_rows; validate every source row without dropping any."""
    if max_rows < 1:
        raise ValueError("max-rows must be positive")
    start_stamp = file_stamp(source)
    source_hash = hashlib.sha256()
    seen: set[str] = set()
    selected: list[tuple[int, int, dict[str, Any]]] = []
    with source.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            source_hash.update(line)
            if not line.strip():
                raise ValueError(f"blank JSONL row at line {line_number}; no silent row drops")
            row = json.loads(line)
            identifier, _, _, _ = validate_wts(row)
            if identifier in seen:
                raise ValueError(f"duplicate WTS id {identifier!r}")
            seen.add(identifier)
            priority = int(json_digest({"seed": seed, "id": identifier, "row": row}), 16)
            candidate = (-priority, -line_number, row)
            if len(selected) < max_rows:
                heapq.heappush(selected, candidate)
            elif candidate[:2] > selected[0][:2]:
                heapq.heapreplace(selected, candidate)
    if not seen:
        raise ValueError("WTS JSONL source is empty")
    if file_stamp(source) != start_stamp:
        raise RuntimeError("data source changed during selection")
    rows = sorted((-negative_line, row) for _, negative_line, row in selected)
    metadata = {
        "protocol": PROTOCOL,
        "source": str(source),
        "source_sha256": source_hash.hexdigest(),
        "source_rows": len(seen),
        "requested_max_rows": max_rows,
        "selected_rows": len(rows),
        "selection_seed": seed,
        "selection_rule": "lowest_sha256_canonical_json_seed_id_raw_row_then_source_line_v1",
        "scoring_order": "source_line_ascending",
    }
    return rows, metadata


def answer_alignment(
    tokenizer: Any,
    input_ids: list[int],
    labels: list[int],
    answer: str,
    *,
    choices: tuple[str, ...] = LETTERS,
) -> tuple[int, dict[str, int]]:
    validate_choices(choices)
    if answer not in choices:
        raise ValueError("gold answer must belong to the actual question choices")
    if len(input_ids) != len(labels):
        raise ValueError("input and label lengths disagree")
    supervised = [position for position, label in enumerate(labels) if label != -100]
    if not supervised or supervised[0] < 1:
        raise ValueError("answer must have a supervised token and a preceding prediction position")
    position = supervised[0]
    prefix_ids = input_ids[:position]
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not prefix.endswith("<|im_start|>assistant\n"):
        raise ValueError("first supervised label must immediately follow the native assistant header")
    if tokenizer(prefix, add_special_tokens=False)["input_ids"] != prefix_ids:
        raise ValueError("assistant prefix does not round-trip to the exact expanded input tokens")
    candidates = {}
    for letter in choices:
        encoded = tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
        if len(encoded) != position + 1 or encoded[:position] != prefix_ids:
            raise ValueError(f"candidate {letter} is not exactly one context-stable next token")
        token = encoded[-1]
        if tokenizer.decode([token], skip_special_tokens=False, clean_up_tokenization_spaces=False) != letter:
            raise ValueError(f"candidate {letter} token does not decode to the exact letter")
        candidates[letter] = token
    if len(set(candidates.values())) != len(choices):
        raise ValueError("answer letters must have distinct token IDs")
    if answer not in candidates or labels[position] != candidates[answer] or input_ids[position] != candidates[answer]:
        raise ValueError("first supervised label is not the exact gold answer token")
    return position, candidates


def score_answer(logits: Any, answer_position: int, candidates: dict[str, int], answer: str) -> dict[str, Any]:
    if logits.ndim != 3 or logits.shape[0] != 1 or not 0 < answer_position < logits.shape[1]:
        raise ValueError("expected [1, sequence, vocabulary] logits and a valid answer position")
    # Labels are unshifted: the token at p is predicted by logits at p-1.
    return score_prediction(logits[0, answer_position - 1], answer_position, candidates, answer)


def score_prediction(scores: Any, answer_position: int, candidates: dict[str, int], answer: str) -> dict[str, Any]:
    """Score one full-vocabulary prediction; shared by dense and selected-head layouts."""
    import torch

    if not isinstance(scores, torch.Tensor) or scores.ndim != 1 or answer_position < 1:
        raise ValueError("expected full-vocabulary prediction scores and a positive answer position")
    choices = tuple(candidates)
    validate_choices(choices)
    if len(set(candidates.values())) != len(choices) or answer not in candidates:
        raise ValueError("expected distinct ordered answer token IDs and gold among actual choices")
    scores = scores.detach().float()
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite answer-position logits")
    if any(not 0 <= token < scores.shape[0] for token in candidates.values()):
        raise ValueError("answer token is outside model vocabulary")
    ids = torch.tensor(list(candidates.values()), device=scores.device)
    choice_scores = scores[ids]
    best_choice_index = int(choice_scores.argmax())
    choice_prediction = choices[best_choice_index]
    first_token = int(scores.argmax())
    reverse = {token: letter for letter, token in candidates.items()}
    gold_id = candidates[answer]
    return {
        "gold": answer,
        "candidate_token_ids": candidates,
        "answer_token_position": answer_position,
        "prediction_logit_position": answer_position - 1,
        "restricted_choice_prediction": choice_prediction,
        "restricted_choice_correct": choice_prediction == answer,
        "strict_first_token_id": first_token,
        "strict_first_token_letter": reverse.get(first_token),
        "strict_first_token_correct": first_token == gold_id,
        "valid_first_letter": first_token in reverse,
        "answer_nll": float(torch.logsumexp(scores, dim=0) - scores[gold_id]),
        "restricted_choice_nll": float(torch.logsumexp(choice_scores, dim=0) - scores[gold_id]),
        "candidate_logits": dict(zip(choices, choice_scores.cpu().tolist(), strict=True)),
        "restricted_choice_tied_max": int((choice_scores == choice_scores.max()).sum()) > 1,
        "strict_first_token_tied_max": int((scores == scores.max()).sum()) > 1,
    }


def evaluate_row(model: Any, packer: Any, objective: Any, row: dict[str, Any]) -> dict[str, Any]:
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.roles.trainer.validation.sft import (
        evaluate_sft_batch,
    )
    from cosmos_framework.posttrain.schema import (
        ObsBundle,
        SFTTarget,
        Trajectory,
        Transition,
    )

    _, answer, choices, _ = validate_wts(row)
    target = SFTTarget(
        messages=row["messages"],
        tools=row.get("tools"),
        supervise=row.get("supervise", "all_assistant"),
        opener=row.get("opener", "self"),
    )
    captured: dict[str, Any] = {}
    collate = packer.sft_collate_fn

    def observe_batch(samples, *args, **kwargs):
        import torch

        if len(samples) != 1 or "inputs" in captured:
            raise RuntimeError("heldout scorer requires exactly one native collation per row")
        batch = collate(samples, *args, **kwargs)
        position, candidates = answer_alignment(
            packer.tokenizer,
            samples[0].input_ids,
            samples[0].labels,
            answer,
            choices=choices,
        )
        valid = (
            (batch["labels"][:, 1:] != -100)
            & batch["attention_mask"][:, 1:].bool()
            & batch["attention_mask"][:, :-1].bool()
        )
        selected_positions = torch.nonzero(valid[0]).flatten()
        answer_rows = torch.nonzero(selected_positions == position - 1).flatten()
        if answer_rows.numel() != 1:
            raise ValueError("gold answer prediction must occur exactly once in the native supervised selection")
        captured.update(
            position=position,
            candidates=candidates,
            inputs={name: tensor_digest(value.contiguous()) for name, value in sorted(batch.items())},
            sequence_length=len(samples[0].input_ids),
            supervised_tokens=int(valid.sum()),
            answer_selected_row=int(answer_rows.item()),
        )
        if "pixel_values_videos" not in batch or not batch["pixel_values_videos"].numel():
            raise RuntimeError("native heldout row lost actual video conditioning")
        return batch

    def observe_inputs(_module, _args, kwargs):
        if "forward_seen" in captured or "inputs" not in captured:
            raise RuntimeError("heldout scorer requires exactly one native forward after collation")
        for name, expected in captured["inputs"].items():
            if name != "labels" and (name not in kwargs or tensor_digest(kwargs[name].contiguous()) != expected):
                raise RuntimeError(f"native forward changed the observed {name} input")
        captured["forward_seen"] = True

    def observe_forward(_module, _args, kwargs, output):
        observe_inputs(_module, _args, kwargs)
        import torch

        logits = output if isinstance(output, torch.Tensor) else output.logits
        captured["scores"] = score_answer(logits, captured["position"], captured["candidates"], answer)

    def observe_selected_head(_module, _args, output):
        import torch

        if not captured.get("forward_seen") or "scores" in captured:
            raise RuntimeError("selected head must execute once after the observed native backbone")
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != captured["supervised_tokens"]:
            raise ValueError("selected head must return [supervised tokens, full vocabulary] logits")
        captured["scores"] = score_prediction(
            output[captured["answer_selected_row"]], captured["position"], captured["candidates"], answer
        )

    packer.sft_collate_fn = observe_batch
    handles = []
    try:
        if getattr(model, "sft_logits_mode", "full") == "supervised":
            handles.append(model.net.model.register_forward_pre_hook(observe_inputs, with_kwargs=True))
            handles.append(model.net.lm_head.register_forward_hook(observe_selected_head))
        else:
            handles.append(model.net.register_forward_hook(observe_forward, with_kwargs=True))
        metrics = evaluate_sft_batch(
            model,
            objective,
            [Trajectory(transitions=[Transition(obs=ObsBundle(), targets=target)])],
        )
    finally:
        for handle in handles:
            handle.remove()
        packer.sft_collate_fn = collate
    loss, count = float(metrics["sft_loss"]), int(metrics["sft_valid_tokens"])
    if not math.isfinite(loss) or count < 1 or count != captured["supervised_tokens"]:
        raise RuntimeError("native SFT loss/token count is invalid or differs from observed supervision")
    return {
        **captured["scores"],
        "native_sft_ce": loss,
        "native_supervised_tokens": count,
        "sequence_length": captured["sequence_length"],
        "inputs": captured["inputs"],
        "input_sha256": json_digest(captured["inputs"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate zero rows")
    count = len(rows)
    tokens = sum(row["native_supervised_tokens"] for row in rows)
    return {
        "rows": count,
        "native_supervised_tokens": tokens,
        "restricted_choice_accuracy": sum(row["restricted_choice_correct"] for row in rows) / count,
        "strict_first_token_accuracy": sum(row["strict_first_token_correct"] for row in rows) / count,
        "valid_letter_rate": sum(row["valid_first_letter"] for row in rows) / count,
        "token_weighted_ce": math.fsum(row["native_sft_ce"] * row["native_supervised_tokens"] for row in rows) / tokens,
        "answer_nll": math.fsum(row["answer_nll"] for row in rows) / count,
        "restricted_choice_nll": math.fsum(row["restricted_choice_nll"] for row in rows) / count,
        "restricted_choice_tied_rows": sum(row["restricted_choice_tied_max"] for row in rows),
        "strict_first_token_tied_rows": sum(row["strict_first_token_tied_max"] for row in rows),
        "unique_exact_video_content_clusters": len({row["media_sha256"] for row in rows}),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("WORLD_SIZE") != "1" or os.environ.get("RANK") != "0" or os.environ.get("LOCAL_RANK") != "0":
        raise ValueError("launch with torchrun --standalone --nproc-per-node=1; only rank zero/world one is supported")
    receipt = {
        name: os.environ.get(name, "") for name in ("COSMOS_TRAINING_CODE_REVISION", "COSMOS_TRAINING_WHEEL_SHA256")
    }
    for name, length in (
        ("COSMOS_TRAINING_CODE_REVISION", 40),
        ("COSMOS_TRAINING_WHEEL_SHA256", 64),
    ):
        if not re.fullmatch(rf"[0-9a-f]{{{length}}}", receipt[name]):
            raise ValueError(f"{name} must identify the installed sealed build")
    source, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    source_stamp = file_stamp(source)
    raw_rows, selection = select_scoring_rows(args, source)

    from cosmos_framework.posttrain.data.datasets.sft import normalize_row, row_identity
    from cosmos_framework.posttrain.tests.qlora_integration import (
        _json_digest,
        _restore_probe_execution,
        _write_json,
    )

    media: dict[str, dict[str, Any]] = {}

    def media_digest(path: str) -> str:
        file = Path(path)
        if path not in media:
            before = file_stamp(file)
            digest = file_digest(file)
            if file_stamp(file) != before:
                raise RuntimeError("video changed while hashing")
            media[path] = {"sha256": digest, "stamp": before}
        if file_stamp(file) != media[path]["stamp"]:
            raise RuntimeError("selected video changed after selection")
        return media[path]["sha256"]

    rows, frozen = [], []
    for line, raw in raw_rows:
        row = normalize_row(raw, source.parent, label=f"{source}:{line}")
        identifier, answer, choices, video = validate_wts(row)
        record = {
            "source_line": line,
            "id": identifier,
            "row_identity": row_identity(row, media_digest=media_digest)[0],
            "raw_row_sha256": json_digest(raw),
            "normalized_row_sha256": json_digest(row),
            "gold": answer,
            "choices": choices,
            "media_sha256": media_digest(video),
            "media_size_bytes": media[video]["stamp"][2],
            "media_path": video,
            "row": row,
        }
        rows.append(row)
        frozen.append(record)
    if "sealed_rows" in selection:
        for actual, expected in zip(frozen, selection.pop("sealed_rows"), strict=True):
            if any(actual[key] != expected[key] for key in ("id", "raw_row_sha256", "media_sha256")):
                raise ValueError("actual selected media differs from sealed split")
    selection["ordered_row_identities"] = [record["row_identity"] for record in frozen]
    selection["ordered_ids"] = [record["id"] for record in frozen]
    # Content-only key supports matching the same selected examples across relocated media roots.
    selection["selected_content_sha256"] = _json_digest(
        [{key: record[key] for key in ("id", "row_identity", "gold", "choices", "media_sha256")} for record in frozen]
    )
    with (output / "selected_rows.jsonl").open("x", encoding="utf-8") as stream:
        for record in frozen:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    selection["selected_rows_file_sha256"] = file_digest(output / "selected_rows.jsonl")
    _write_json(output / "selection.json", selection)
    # All rows and all source/media identities above are frozen before any score or model construction.
    import torch

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.checkpoint.adapter import (
        load_adapter_weights,
        read_adapter_metadata,
    )
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
    from cosmos_framework.posttrain.config.build import build_backbone, build_optimizer
    from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
    from cosmos_framework.posttrain.entrypoints.sft import (
        sft_policy_source,
        validate_config,
    )
    from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
    from cosmos_framework.posttrain.utils.adapter_identity import (
        adapter_processor_identity,
    )
    from cosmos_framework.posttrain.utils.determinism import enable_determinism
    from cosmos_framework.posttrain.utils.dist import (
        destroy_process_group_if_initialized,
        init_local_process_group,
    )
    from cosmos_framework.posttrain.utils.identity import canonical_config_value
    from cosmos_framework.posttrain.utils.native_compile import native_compile_diagnostics
    from cosmos_framework.utils import misc
    from cosmos_framework.utils.config import load_config

    config = load_config(args.config, ["--", *args.override])
    saved = read_adapter_metadata(args.checkpoint) if args.checkpoint else None
    if saved is not None:
        _restore_probe_execution(config, saved["identity"], args.override)
    rl = validate_config(config)
    if not rl.deterministic or rl.ce_exponent != 0 or rl.moe_lbl_coeff != 0:
        raise ValueError("heldout comparison requires deterministic=True and ordinary dense token-mean CE")
    enable_determinism(strict=True)
    init_local_process_group()
    try:
        misc.set_random_seed(rl.seed, by_rank=False)
        model_source, trust = sft_policy_source(config)
        packer = build_sft_packer(
            model_source,
            max_length=rl.max_length,
            processor_kwargs=rl.processor_kwargs,
            trust_remote_code=trust,
            mask_think_opener=rl.mask_think_opener,
        )
        model = build_backbone(config)
        optimizer = build_optimizer(model.net, rl.optimizer)
        codec = model.net._cosmos_adapter_checkpoint_codec
        codec.bind_processor_identity(
            adapter_processor_identity(packer, source_path=model_source, source_identity=codec.identity["base"])
        )
        codec.assert_live(model.net, optimizer)
        if saved is not None:
            load_adapter_weights(
                args.checkpoint,
                codec,
                manager=CheckpointManager(checkpoint_format="dcp"),
            )
        adapters_before = {name: tensor_digest(parameter) for name, parameter in codec.parameters.items()}
        if saved is None and adapters_before != codec.identity["initial_adapters"]:
            raise RuntimeError("frozen-baseline arm must use exact initial adapters")
        model.net.eval()
        objective = make_objective(
            packer,
            ce_exponent=rl.ce_exponent,
            moe_lbl_coeff=rl.moe_lbl_coeff,
            moe_lbl_method=rl.moe_lbl_method,
            serve_contract=rl.serve_contract,
        )
        identity = {key: value for key, value in codec.identity.items() if key != "training_contract"}
        _write_json(output / "model_identity.json", identity)
        report_rows = []
        with (output / "scores.jsonl").open("x", encoding="utf-8") as stream:
            for index, (row, selected) in enumerate(zip(rows, frozen, strict=True), 1):
                media_digest(selected["media_path"])
                scores = evaluate_row(model, packer, objective, row)
                media_digest(selected["media_path"])
                record = {key: value for key, value in selected.items() if key != "row"}
                record.update(scores)
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                report_rows.append(record)
                if index == 1 or index % 16 == 0 or index == len(rows):
                    print(
                        json.dumps(
                            {
                                "event": "heldout_progress",
                                "completed": index,
                                "total": len(rows),
                            }
                        ),
                        flush=True,
                    )
            os.fsync(stream.fileno())
        codec.validate_frozen_base()
        if adapters_before != {name: tensor_digest(parameter) for name, parameter in codec.parameters.items()}:
            raise RuntimeError("evaluation changed adapter tensors")
        if file_stamp(source) != source_stamp or file_digest(source) != selection["source_sha256"]:
            raise RuntimeError("data source changed during evaluation")
        for path in media:
            media_digest(path)
        report = {
            "protocol": PROTOCOL,
            "complete": True,
            "arm": args.arm,
            "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
            "adapter_state": "reloaded_checkpoint" if saved is not None else "initial_zero_B_frozen_base",
            "metrics": aggregate(report_rows),
            "selection_sha256": _json_digest(selection),
            "selected_content_sha256": selection["selected_content_sha256"],
            "scores_sha256": file_digest(output / "scores.jsonl"),
            "model_identity_sha256": _json_digest(identity),
            "adapter_tensors_sha256": _json_digest(adapters_before),
            "processor_contract": codec.identity["processor_contract"],
            "objective_identity": objective.identity,
            "quantized_matrices": len(identity["quantized_tensors"]),
            "recipe": args.config,
            "overrides": args.override,
            "model_seed": rl.seed,
            "runtime_precision": model.net._cosmos_adapter_runtime_precision,
            "compile_diagnostics": native_compile_diagnostics(model.net.modules()),
            "max_length": rl.max_length,
            "processor_kwargs": canonical_config_value(rl.processor_kwargs),
            "build_receipt": receipt,
            "evaluator_sha256": file_digest(Path(__file__).resolve()),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes")},
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "world_size": 1,
            "elapsed_seconds": time.monotonic() - start,
            "metric_contract": {
                "restricted_choice": "argmax over the actual question's two to four contiguous A/B[/C[/D]] token logits at the first assistant answer position",
                "strict_first_token": "full-vocabulary argmax equals the gold letter token",
                "valid_letter_rate": "full-vocabulary argmax belongs to the actual question choices",
                "answer_nll": "mean full-vocabulary negative log probability of the single gold answer token",
                "token_weighted_ce": "native supervised-token CE; includes assistant end token and preserves existing masks",
                "tie_break": "torch.argmax: first actual choice in canonical A/B/C/D order, lowest full-vocabulary token id",
                "media_clusters": "exact video content SHA256 only; related camera views/events may need broader grouping",
                "limitations": "bounded teacher-forced diagnostic; not free generation or full heldout/release qualification",
            },
        }
        _write_json(output / "summary.json", report)
        return {"output": str(output), "metrics": report["metrics"], "complete": True}
    finally:
        destroy_process_group_if_initialized()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        help="committed DCP or portable adapter; omit for the frozen-base arm",
    )
    parser.add_argument("--data", required=True, help="converted native WTS validation JSONL")
    parser.add_argument(
        "--output",
        required=True,
        help="new output directory; existing directories are refused",
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--max-rows", "--maxrows", dest="max_rows", type=int, default=256)
    selection_group.add_argument("--all-rows", action="store_true", help="score every row from a sealed group split")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split-manifest-sha256")
    parser.add_argument("--split", choices=("development", "regression", "untouched_diagnostic", "final"))
    parser.add_argument("--selection-receipt")
    parser.add_argument("--selection-receipt-sha256")
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="subset seed, independent of model/recipe seed",
    )
    parser.add_argument(
        "--arm",
        default="unspecified",
        help="descriptive label such as frozen_nf4 or trained_nf4",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="repeatable native config dotlist override",
    )
    args = parser.parse_args()
    output_existed = Path(args.output).resolve().exists()
    try:
        print(json.dumps(run(args), sort_keys=True, allow_nan=False), flush=True)
    except Exception as error:
        output = Path(args.output).resolve()
        if not output_existed and output.is_dir() and not (output / "failure.json").exists():
            write_json(
                output / "failure.json",
                {
                    "complete": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    main()
