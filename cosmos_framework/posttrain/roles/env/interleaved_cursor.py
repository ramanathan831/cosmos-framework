# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact-exposure interleaving inside the existing seeded cursor permutation."""

from collections import deque
from copy import deepcopy

from cosmos_framework.posttrain.roles.env.cursor import IndexedCursor
from cosmos_framework.posttrain.utils.identity import canonical_json_sha256


class InterleavedCursor(IndexedCursor):
    """Single-rank binary proportional interleaving, with a distinct state format.

    Every window retains the original permutation's members and within-group
    order. Integer deficits spread groups proportionally to their actual counts;
    homogeneous windows and incomplete tails remain valid. This changes update
    order, never exposure or objective weights. Global RNG is unused.
    """

    def __init__(self, dataset, *, id_prefix, window, **kwargs):
        if not isinstance(id_prefix, str) or not id_prefix or type(window) is not int or window < 2:
            raise ValueError("interleaving requires a nonempty ID prefix and integer window >=2")
        if kwargs.get("shard_count", 1) != 1 or kwargs.get("shard_index", 0) != 0:
            raise ValueError("SFT interleaving currently supports one data rank only")
        super().__init__(dataset, **kwargs)
        ids = [dataset[index].get("id") for index in range(len(dataset))]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("interleaving requires nonempty string row IDs")
        self._groups = [int(value.startswith(id_prefix)) for value in ids]
        if set(self._groups) != {0, 1}:
            raise ValueError("interleaving requires both matching and nonmatching IDs")
        self._window = window
        self._policy = {
            "kind": "proportional_binary_windows_v1",
            "id_prefix": id_prefix,
            "window": window,
            "groups_sha256": canonical_json_sha256(self._groups, context="SFT interleaving groups"),
        }

    def _interleave(self, order):
        result = []
        for start in range(0, len(order), self._window):
            block = order[start : start + self._window]
            queues = [deque(i for i in block if self._groups[i] == group) for group in (0, 1)]
            counts, used = [len(q) for q in queues], [0, 0]
            for position in range(len(block)):
                available = [group for group in (0, 1) if queues[group]]
                # Stable group-zero tie break, recorded by the policy version.
                group = max(available, key=lambda g: (position + 1) * counts[g] - used[g] * len(block))
                result.append(queues[group].popleft())
                used[group] += 1
        return result

    def _rebuild_order(self):
        super()._rebuild_order()
        self._order = self._interleave(self._order)

    def state_dict(self):
        # The ordinary cursor rejects version2, preventing an unrecorded switch
        # back to untransformed order. Ordinary version1 checkpoints stay intact.
        return {**super().state_dict(), "version": 2, "interleave": deepcopy(self._policy)}

    def ordering_identity(self):
        return deepcopy(self._policy)

    def load_state_dict(self, state):
        if self._cursor_resume == "off":
            return False
        policy = state.get("interleave") if isinstance(state, dict) else None
        if (
            not isinstance(state, dict)
            or type(state.get("version")) is not int
            or state["version"] != 2
            or not isinstance(policy, dict)
            or type(policy.get("window")) is not int
            or policy != self._policy
        ):
            raise ValueError("exact SFT resume requires the same interleaving policy and cursor version2")
        native = {k: v for k, v in state.items() if k != "interleave"}
        native["version"] = 1
        result = super().load_state_dict(native)
        self._order = self._interleave(self._order)
        return result
