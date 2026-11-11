import pytest
import torch

from cosmos_framework.posttrain.tests.helpers.sft_objective import Policy
from tools.quantized_optimizers import late_observer


@pytest.mark.parametrize("failure", [False, True])
def test_observer_restores_rng_and_mixed_modes_on_failure(monkeypatch, failure):
    policy = Policy()
    policy.net.dropout.eval()
    before = torch.get_rng_state().clone()

    def observed(*args):
        if failure:
            torch.rand(1)
            policy.net.train()
            raise ValueError("injected failure")
        return {"native_tokens": 1346}

    monkeypatch.setattr(late_observer, "substantive_batch_report", observed)
    if failure:
        with pytest.raises(ValueError, match="injected"):
            late_observer.observe(policy, None, None, None)
    else:
        assert late_observer.observe(policy, None, None, None) == {"native_tokens": 1346}
    assert torch.equal(before, torch.get_rng_state())
    assert policy.net.training and not policy.net.dropout.training


def test_observer_rejects_parameter_mutation(monkeypatch):
    policy = Policy()

    def mutated(*args):
        with torch.no_grad():
            next(policy.net.parameters()).add_(1)
        return {}

    monkeypatch.setattr(late_observer, "substantive_batch_report", mutated)
    with pytest.raises(RuntimeError, match="mutated"):
        late_observer.observe(policy, None, None, None)
