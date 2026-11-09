"""Reconstruct the existing seed43 batches; no training or candidate selection."""

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor

ROOT = Path(__file__).parent


def main():
    metadata = json.loads(gzip.decompress((ROOT / "training_metadata.json.gz").read_bytes()))
    saved = json.loads((ROOT / "seed43_cursor.json").read_text())["states"]["0"]
    old = json.loads(
        gzip.decompress((ROOT.parent / "quantized_optimizers_round5/cold_replacement_raw.json.gz").read_bytes())
    )
    manifest_text = old["trial0/manifest.json"]
    assert hashlib.sha256(manifest_text.encode()).hexdigest() == metadata["reference_manifest_sha256"]
    manifest = json.loads(manifest_text)
    assert saved["dataset_fingerprint"] in manifest["sources"]
    for index, row in enumerate(metadata["rows"]):
        assert row["index"] == index
        entry = manifest["entries"][row["target_key"]]
        assert all(row[key] == entry[key] for key in ("sequence_length", "supervised_tokens"))
    assert saved["seed"] == 43 and saved["dataset_len"] == len(metadata["rows"]) == 346
    assert saved["position"] == 60 and saved["epoch"] == 1 and saved["world_size"] == 1

    class Rows:
        def __len__(self):
            return len(metadata["rows"])

        def __getitem__(self, index):
            return metadata["rows"][index]

        def cursor_fingerprint(self):
            return saved["dataset_fingerprint"]

    dataset = Rows()
    cursor = IndexedCursor(dataset, shuffle=True, seed=43, cursor_resume="strict")
    batches = []
    for step in range(1, 16):
        rows = [dataset[cursor.next_index()] for _ in range(4)]
        weight = {}
        for exponent in [1.0, 0.975, 0.95]:
            raw = [r["supervised_tokens"] ** (1 - exponent) for r in rows]
            weight[str(exponent)] = [w / sum(raw) for w in raw]
        batches.append(
            dict(
                step=step,
                rows=rows,
                row_mean_weights=weight,
                text_rows=sum(r["kind"] == "text" for r in rows),
                text_weight_975=sum(w for w, r in zip(weight["0.975"], rows) if r["kind"] == "text"),
            )
        )
    assert cursor.state_dict() == saved
    windows = {}
    for start in [0, 5, 10]:
        rows = [r for batch in batches[start : start + 5] for r in batch["rows"]]
        windows[f"{start + 1}-{start + 5}"] = dict(
            kinds=dict(Counter(r["kind"] for r in rows)),
            recording_group_counts=dict(Counter(r["group"] for r in rows)),
            supervised_tokens=sum(r["supervised_tokens"] for r in rows),
            sequence_lengths=[r["sequence_length"] for r in rows],
            mixed_objective_updates=sum(0 < b["text_rows"] < 4 for b in batches[start : start + 5]),
        )
    result = dict(
        scope="Existing seed43 training order only; no new trial or causal claim.",
        source_sha256=metadata["source_sha256"],
        reference_manifest_sha256=metadata["reference_manifest_sha256"],
        original_and_weight975_data_pt_sha256="91a1cdd56ff302e3c6108c6a7b6ae53327b1d6e29ba1cfe4fc984437742d0bca",
        exact_native_cursor_at_update15=True,
        saved_cursor=saved,
        windows=windows,
        batches=[
            {
                **{key: value for key, value in batch.items() if key != "rows"},
                "row_indices": [row["index"] for row in batch["rows"]],
                "supervised_tokens": [row["supervised_tokens"] for row in batch["rows"]],
            }
            for batch in batches
        ],
        raw_metadata="training_metadata.json.gz",
        interpretation="The same60rows and cursor are used in both arms. Exponent changes row-mean weighting within each batch, not labels or evaluation reduction. These counts localize which steps can differ; they do not identify the cause of a later validation-loss change.",
    )
    (ROOT / "training_order_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(windows, indent=2))


if __name__ == "__main__":
    main()
