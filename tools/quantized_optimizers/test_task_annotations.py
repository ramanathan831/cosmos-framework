"""The raw evaluation instruction is not part of an annotated answer."""

import pytest

from tools.quantized_optimizers.prepare_development import decode_annotation

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("escaped", [False, True])
def test_final_option_does_not_absorb_direct_answer_instruction(escaped):
    question = (
        "<video>\nWhat is the weather?\nA: Clear\nB: Rainy\nC: Snowy\nD: Cloudy\n\n"
        "Answer with the option's letter from the given choices directly."
    )
    if escaped:
        question = question.replace("\n", "\\n")
    row = {"conversations": [{"value": question}, {"value": "D"}]}
    q, options, gold = decode_annotation(row)
    assert q == "What is the weather?"
    assert options[gold] == "Cloudy"
    assert list(options) == ["A", "B", "C", "D"]


def test_training_annotation_without_footer_keeps_answer_text():
    row = {"conversations": [{"value": "<video>\nQ?\nA: Two-way traffic\nB: One-way traffic"}, {"value": "A"}]}
    assert decode_annotation(row) == ("Q?", {"A": "Two-way traffic", "B": "One-way traffic"}, "A")


def test_bad_gold_mapping_is_rejected():
    row = {"conversations": [{"value": "Q?\nA: Yes\nB: No"}, {"value": "D"}]}
    with pytest.raises(ValueError, match="gold letter"):
        decode_annotation(row)
