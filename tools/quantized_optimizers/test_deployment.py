"""Meaningful control checks for dense qualification and byte-level identities."""

import pytest
import torch

from tools.quantized_optimizers.score_dense import parse_options, score_answer
from tools.quantized_optimizers.state_digest import digest

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_dense_answer_uses_previous_position_and_full_vocabulary_nll():
    logits = torch.full((1, 4, 8), -4.0)
    logits[0, 1, 6] = 5.0  # gold is at input position2
    logits[0, 2, 7] = 9.0  # using unshifted logits would predict the wrong answer
    result = score_answer(logits, 2, {"A": 6, "B": 7}, "A")
    assert result["strict_first_token_correct"] and result["restricted_choice_correct"]
    assert result["answer_nll"] == pytest.approx(float(-logits[0, 1].log_softmax(-1)[6]))
    assert result["prediction_logit_position"] == 1


@pytest.mark.parametrize("question", ["Q\nA: yes\nC: no", "Q\nA: yes\nB:", "Q\nA: yes\nB. no"])
def test_malformed_task_choices_rejected(question):
    with pytest.raises(ValueError):
        parse_options(question)


def test_state_digest_ignores_storage_pointers_but_preserves_integer_bits_dtype_and_scalar():
    a = {"moment": torch.tensor([0, 128, 255], dtype=torch.uint8), "step": 7}
    b = {"moment": a["moment"].clone(), "step": 7}
    assert digest(a) == digest(b)
    b["moment"][1] = 127
    assert digest(a) != digest(b)
    assert digest(a) != digest({"moment": a["moment"].to(torch.int16), "step": 7})
    assert digest(a) != digest({"moment": a["moment"], "step": 8})


def test_strict_determinism_requests_kernels_and_keeps_legacy_default(monkeypatch):
    from cosmos_framework.posttrain.config.sft import SFTConfig
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    with pytest.raises(ValueError, match="requires deterministic"):
        SFTConfig(strict_determinism=True)
    with pytest.raises(TypeError, match="must be bool"):
        SFTConfig(strict_determinism="true")
    assert SFTConfig(deterministic=True, strict_determinism=True).strict_determinism
    old = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    try:
        enable_determinism()
        assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
        enable_determinism(strict=True)
        assert (
            torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        )
    finally:
        torch.use_deterministic_algorithms(old[0], warn_only=old[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = old[2:]


def test_recovery_probe_records_each_clock_once_and_checks_nested_step_agreement(monkeypatch):
    """CPU control test only; dense CUDA continuation is a separate execution gate."""
    from types import SimpleNamespace

    from tools.quantized_optimizers import prepared_canary as canary

    # Initialize the parent wrapper as happens during the real backend preflight.
    torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])

    class WrappedAdamW(torch.optim.AdamW):
        mutate_after_inner = False

        @torch.no_grad()
        def step(self, closure=None):
            result = super().step(closure)
            if self.mutate_after_inner:
                self.param_groups[0]["params"][0].add_(1)
            return result

    net = torch.nn.Linear(3, 2, bias=False)
    optimizer = WrappedAdamW(net.parameters())
    monkeypatch.setattr(canary, "captured", {})
    monkeypatch.setattr(canary, "original", lambda *_args: SimpleNamespace(_net=lambda: net, opt=optimizer))
    monkeypatch.setenv("RECOVERY_ARM", "adamw")
    canary.capture(None, None, None, None, None)
    try:
        for _ in range(4):
            net.weight.grad = torch.ones_like(net.weight)
            optimizer.step()
        records = canary.captured["update_probes"]
        assert [row["step"] for row in records] == [1, 2, 3, 4]
        assert len({row["sha256"] for row in records}) == 4
        optimizer.mutate_after_inner = True
        with pytest.raises(RuntimeError, match="after its inner update"):
            optimizer.step()
    finally:
        canary.captured["probe_handle"].remove()


def test_completed_recovery_state_resolves_native_worker_handle(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.workers.worker_group import Handle
    from tools.quantized_optimizers import prepared_canary as canary

    net = torch.nn.Linear(3, 2, bias=False)
    opt = torch.optim.AdamW(net.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    cursor = {"states": {"0": {"epoch": 2, "position": 3}}}
    data = SimpleNamespace(state_dict=lambda: Handle(value=cursor))
    trainer = SimpleNamespace(
        _net=lambda: net, opt=opt, lr_scheduler=scheduler, _update_state_dict=lambda: {"optimizer_step": 41}
    )
    monkeypatch.setattr(canary, "captured", {"trainer": trainer})
    monkeypatch.setattr(canary, "original_step", lambda *_: {"optim/optimizer_step": 41})
    before = digest(torch.optim.Optimizer.state_dict(opt))
    canary.capture_completed_step(data, None, None)
    captured = canary.captured["completed_update_states"][0]
    assert captured["cursor"] == digest(cursor)
    assert captured["optimizer"] == before == digest(torch.optim.Optimizer.state_dict(opt))
    assert captured["step"] == 41
