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
