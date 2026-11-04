"""Extra correlated questions must not create independent bootstrap observations."""

import pytest

from tools.quantized_optimizers.outcome_statistics import paired_quality

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def test_correlated_questions_do_not_shrink_recording_uncertainty():
    def rows(repeats, offset=0):
        return [
            {
                "id": f"{group}-{question}",
                "group": str(group),
                "answer_nll": 1 + group + offset,
                "restricted_choice_correct": group % 2 == 0,
            }
            for group in range(8)
            for question in range(repeats)
        ]

    a = paired_quality([rows(1)] * 3, [rows(1, 0.01 * i) for i in (1, 2, 3)], rows(1), repetitions=500)
    b = paired_quality([rows(4)] * 3, [rows(4, 0.01 * i) for i in (1, 2, 3)], rows(4), repetitions=500)
    assert a["crossed_seed_recording_bootstrap"] == b["crossed_seed_recording_bootstrap"]
    assert not a["guards"]["equal_budget_nll_gain"]
    mismatched = rows(1)
    mismatched[0]["group"] = "wrong"
    with pytest.raises(ValueError, match="identities"):
        paired_quality([mismatched] * 3, [rows(1)] * 3, rows(1))
