"""Temporal attribution after exact replay parity; never final-data selection."""

import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent


def components(row, text, vocabulary):
    pieces = [vocabulary[str(i)].replace("Ġ", " ").replace("Ċ", "\n") for i in row["target_token_ids"]]
    # The pinned observed vocabulary is ASCII plus these space/newline markers.
    # Reject another tokenizer or template rather than infer token boundaries.
    assert all(part.isascii() for part in pieces)
    assert "".join(pieces) == text["answer"] + "<|im_end|>"
    spans, offset = [], 0
    for line, question in zip(text["answer"].splitlines(keepends=True), text["questions"], strict=True):
        prefix = re.match(r"(\d+)\.\s*", line)
        assert prefix and question.startswith(prefix[1] + ". ")
        spans.append((offset, offset + len(line), offset + prefix.end(), question.split(". ", 1)[1]))
        offset += len(line)
    result, offset = [], 0
    for part in pieces:
        if part == "<|im_end|>":
            result.append(("end token", "end token"))
        else:
            matches = [s for s in spans if s[0] <= offset < s[1]]
            assert len(matches) == 1
            start, end, answer_start, question = matches[0]
            assert offset + len(part) <= end
            category = (
                "answer-bearing"
                if part.rstrip("\n") and offset + len(part.rstrip("\n")) > answer_start
                else "format-only"
            )
            result.append((question, category))
        offset += len(part)
    return result


def main():
    raw = json.loads(gzip.decompress((ROOT / "replay_raw.json.gz").read_bytes()))
    assert json.loads(raw["replay_verified.json"])["complete"]
    series = {
        arm: {
            row["step"]: row for row in map(json.loads, raw[arm + "/diagnostics/late_substantive.jsonl"].splitlines())
        }
        for arm in ("original", "weight975")
    }
    order = {
        row["step"]: row
        for row in json.loads((ROOT.parent / "quantized_optimizers_round7/training_order_summary.json").read_text())[
            "batches"
        ]
    }
    observations = []
    prior = {}
    texts = {row["id"]: row for row in json.loads((ROOT / "development_texts.json").read_text())["rows"]}
    vocabulary = json.loads((ROOT / "token_strings.json").read_text())["raw_vocabulary_strings"]
    for step in range(10, 16):
        a, b = series["original"][step], series["weight975"][step]
        assert a["native_tokens"] == b["native_tokens"] == 1346
        assert a["source_fingerprint"] == b["source_fingerprint"]
        tokens = []
        rows = []
        for ba, bb in zip(a["batches"], b["batches"], strict=True):
            assert ba["input_sha256"] == bb["input_sha256"] and ba["input_shapes"] == bb["input_shapes"]
            for ra, rb in zip(ba["rows"], bb["rows"], strict=True):
                assert all(
                    ra[k] == rb[k] for k in ("id", "tokens", "target_token_ids", "trajectory_id", "sequence_length")
                )
                delta = [y - x for x, y in zip(ra["token_losses"], rb["token_losses"], strict=True)]
                parts = components(ra, texts[ra["id"]], vocabulary)
                for position, value in enumerate(delta):
                    key = (ra["id"], position)
                    tokens.append(
                        {
                            "id": ra["id"],
                            "position": position,
                            "token_id": ra["target_token_ids"][position],
                            "token_text": vocabulary[str(ra["target_token_ids"][position])],
                            "question": parts[position][0],
                            "component": parts[position][1],
                            "delta": value,
                            "increment_since_previous": value - prior[key] if key in prior else None,
                            "original_loss": ra["token_losses"][position],
                            "candidate_loss": rb["token_losses"][position],
                        }
                    )
                    prior[key] = value
                rows.append(
                    {
                        "id": ra["id"],
                        "native_mean_delta_contribution": math.fsum(delta) / 1346,
                        "original_nll": ra["token_loss_sum"] / ra["tokens"],
                        "candidate_nll": rb["token_loss_sum"] / rb["tokens"],
                    }
                )
        assert math.isclose(
            math.fsum(t["delta"] for t in tokens) / 1346, b["native_loss"] - a["native_loss"], abs_tol=1e-7
        )
        observation = {
            "step": step,
            "original_native_nll": a["native_loss"],
            "candidate_native_nll": b["native_loss"],
            "ratio": b["native_loss"] / a["native_loss"],
            "delta": b["native_loss"] - a["native_loss"],
            "training_text_rows": order[step]["text_rows"],
            "rows": rows,
            "largest_token_relative_changes": sorted(
                tokens, key=lambda x: abs(x["increment_since_previous"] or 0), reverse=True
            )[:30],
        }
        grouped = defaultdict(lambda: {"tokens": 0, "delta": 0.0, "increment": 0.0})
        for token in tokens:
            key = token["question"] + " / " + token["component"]
            grouped[key]["tokens"] += 1
            grouped[key]["delta"] += token["delta"] / 1346
            grouped[key]["increment"] += (token["increment_since_previous"] or 0) / 1346
        observation["question_component_contributions"] = dict(sorted(grouped.items()))
        observations.append(observation)
    output = {
        "complete": True,
        "scope": "One controlled fused seed43 replay; exact native batch-four teacher-forced text loss, not free generation or optimizer-specific evidence.",
        "parity": json.loads(raw["replay_verified.json"]),
        "updates": observations,
        "interpretation_limit": "Temporal contributions localize where the recipe effect changes; they do not isolate a sole causal role for batch composition versus inherited parameters/moments.",
    }
    (ROOT / "replay_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in x.items()
                    if k not in ("rows", "largest_token_relative_changes", "question_component_contributions")
                }
                for x in observations
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
