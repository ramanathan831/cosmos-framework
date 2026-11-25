# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real multiprocess DCP publication control checks; no accelerator qualification."""

from pathlib import Path

import pytest
import torch


def _publish(rank, directory, divergent_recipe, stale_client_stat):
    import os
    from types import SimpleNamespace

    import torch.distributed as dist

    from cosmos_framework.posttrain.checkpoint import CheckpointManager
    from cosmos_framework.posttrain.control.validation.lifecycle import ValidationTrigger
    from cosmos_framework.posttrain.roles.trainer.validation.publication import PublicationJournal
    from cosmos_framework.posttrain.roles.trainer.validation.state import ValidationState
    from cosmos_framework.posttrain.tests.helpers.distributed import init_gloo_process_group

    init_gloo_process_group(rank, 2, str(Path(directory) / "rendezvous"))
    try:
        path = str(Path(directory) / "iter_000004")
        manager = CheckpointManager(checkpoint_format="dcp", keep_optim_last=0, keep_model_last=0)
        state = ValidationState({"enabled": True, "seed": rank if divergent_recipe else 17})
        journal = PublicationJournal(state)
        state.begin(4, frozenset({ValidationTrigger.PERIODIC}))

        def control():
            return {
                "iter": 4,
                "validation": state.state_dict(version=2, extra={"publication": journal.checkpoint_record()}),
            }

        # Exceed the inventory's small-file content-hashing threshold. Replicated
        # model and rank-owned optimizer leaves exercise both writer layouts.
        manager.save_dcp(
            path,
            {"model": {"weight": torch.ones(2**21)}, "optim": {str(rank): torch.zeros(2**21)}},
            {"trainer": control()},
            async_safe_snapshot=False,
            publish=False,
        )
        before = manager.checkpoint_provenance(path)
        journal.complete(4)
        if divergent_recipe:
            with pytest.raises(RuntimeError, match="publication recipe identity differs across ranks"):
                journal.publish(
                    path, manager=manager, publication_identity={}, trainer_state=control, on_checkpoint=lambda: None
                )
            assert manager.checkpoint_provenance(path) == before
        else:
            with pytest.MonkeyPatch.context() as patch:
                if stale_client_stat and rank == 1:
                    import importlib

                    from cosmos_framework.posttrain.checkpoint.synchronized_stat import synchronized_stat

                    target = str(Path(path) / "model" / "__0_0.distcp")
                    original = os.stat

                    def stale(file, *args, **kwargs):
                        result = original(file, *args, **kwargs)
                        if not isinstance(file, int) and os.fspath(file) == target:
                            return SimpleNamespace(
                                st_size=result.st_size,
                                st_mode=result.st_mode,
                                st_dev=result.st_dev,
                                st_ino=result.st_ino,
                                st_mtime_ns=result.st_mtime_ns - 2_000_000_000,
                                st_ctime_ns=result.st_ctime_ns - 2_000_000_000,
                            )
                        return result

                    if stale_client_stat == "server":
                        provenance = importlib.import_module("cosmos_framework.posttrain.checkpoint.provenance")
                        target_attributes = synchronized_stat(target)

                        def stale_server(file):
                            result = synchronized_stat(file)
                            if (result.st_dev, result.st_ino) == (
                                target_attributes.st_dev,
                                target_attributes.st_ino,
                            ):
                                result.st_mtime_ns -= 1_000_000_000
                                result.st_ctime_ns -= 1_000_000_000
                            return result

                        patch.setattr(provenance, "synchronized_stat", stale_server)
                    else:
                        patch.setattr(os, "stat", stale)
                        assert os.stat(target).st_mtime_ns + 2_000_000_000 == synchronized_stat(target).st_mtime_ns
                journal.publish(
                    path, manager=manager, publication_identity={}, trainer_state=control, on_checkpoint=lambda: None
                )
            assert manager.load_component(path, "trainer") == control()
            assert journal.pending_step is None and state.progress.pending_step is None
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "divergent_recipe,stale_client_stat", [(False, False), (True, False), (False, True), (False, "server")]
)
def test_large_dcp_validation_publication_across_processes(
    tmp_path, monkeypatch, divergent_recipe, stale_client_stat
):
    if stale_client_stat:
        # Ranks disagreeing about cached attributes need the opt-in shared-filesystem identity.
        monkeypatch.setenv("COSMOS_CHECKPOINT_SHARED_FS_IDENTITY", "1")
    torch.multiprocessing.spawn(
        _publish, args=(str(tmp_path), divergent_recipe, stale_client_stat), nprocs=2, join=True
    )


@pytest.mark.L0
@pytest.mark.CPU
def test_inventory_diagnostics_preserve_signature_and_identify_changed_entry(tmp_path):
    from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature

    tmp_path = tmp_path / "iter_000001"  # the test logger writes console/debug logs into the fixture root
    tmp_path.mkdir()
    small, large = tmp_path / "control", tmp_path / "payload"
    small.write_bytes(b"small control")
    with large.open("wb") as handle:
        handle.truncate(5 * 1024 * 1024)
    baseline = checkpoint_inventory_signature(str(tmp_path))
    before = []
    assert checkpoint_inventory_signature(str(tmp_path), inventory=before) == baseline
    small.write_bytes(b"changed control")
    after = []
    assert checkpoint_inventory_signature(str(tmp_path), inventory=after) != baseline
    assert [a == b for a, b in zip(before, after)] == [True, False, True]
