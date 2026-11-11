"""Read-only native batch-four attribution; never a product timing path."""

from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state, restore_rng_state
from tools.quantized_optimizers.score_dense import substantive_batch_report
from tools.quantized_optimizers.state_digest import digest


def observe(model, packer, objective, dataset):
    rng = collect_rng_state()
    modes = [(module, module.training) for module in model.net.modules()]
    tensors = list(model.net.parameters()) + list(model.net.buffers())
    versions = [tensor._version for tensor in tensors]
    try:
        result = substantive_batch_report(model, packer, objective, dataset, 4)
        if versions != [tensor._version for tensor in tensors]:
            raise RuntimeError("late diagnostic mutated parameters or buffers")
        if digest(collect_rng_state()) != digest(rng) or any(module.training != mode for module, mode in modes):
            raise RuntimeError("native substantive evaluation did not restore RNG/module mode")
        return result
    finally:
        restore_rng_state(rng)
        for module, mode in modes:
            module.training = mode
