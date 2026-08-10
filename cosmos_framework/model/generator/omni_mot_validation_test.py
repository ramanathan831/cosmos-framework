# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import MethodType, SimpleNamespace

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


def test_validation_step_is_deterministic_and_restores_training_cp_state():
    model = object.__new__(OmniMoTModel)
    torch.nn.Module.__init__(model)
    model.net = SimpleNamespace()
    model._cp_local_training_payload = {"kind": "train"}
    model._cp_window_slot = 7
    model._cp_validation_payload = None
    model._cp_validation_window_slot = 0
    model._validation_state_iteration = None
    model._validation_batch_index = 0

    def _training_step(self, _data_batch, _iteration):
        loss = torch.rand(())
        self._cp_local_training_payload = {"kind": "validation"}
        self._cp_window_slot += 1
        return {"batch_size": 1}, loss

    model.training_step = MethodType(_training_step, model)

    _, first_loss = OmniMoTModel.validation_step(model, {}, iteration=100)
    _, second_loss = OmniMoTModel.validation_step(model, {}, iteration=100)
    _, next_pass_first_loss = OmniMoTModel.validation_step(model, {}, iteration=200)

    assert first_loss.item() != second_loss.item()
    assert first_loss.item() == next_pass_first_loss.item()
    assert model._cp_local_training_payload == {"kind": "train"}
    assert model._cp_window_slot == 7
