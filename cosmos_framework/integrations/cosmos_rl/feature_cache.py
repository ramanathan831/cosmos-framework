# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation-only video feature cache shared with the optional Cosmos-RL backend."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from types import MethodType
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class _ValidationVideoFeatureCache:
    """Rank-local, GPU-resident LRU for deterministic validation features."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("validation video feature cache capacity must be positive")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.sync_dummy_encodes = 0
        self.global_all_hit_calls = 0
        self.bypassed_calls = 0

    @staticmethod
    def _grid_tuple(grid_row: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in grid_row.detach().cpu().tolist())

    @staticmethod
    def _distributed_flag(value: bool, device, reduce_op) -> bool:
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            return value
        flag = torch.tensor([int(value)], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=reduce_op)
        return bool(flag.item())

    def get_or_encode(
        self,
        cache_keys,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        encode_fn: Callable,
        spatial_merge_size: int,
    ):
        local_cacheable = (
            video_grid_thw is not None
            and video_grid_thw.ndim == 2
            and isinstance(cache_keys, (list, tuple))
            and len(cache_keys) == int(video_grid_thw.shape[0])
        )
        grids = []
        raw_split_sizes = []
        if local_cacheable:
            grids = [self._grid_tuple(row) for row in video_grid_thw]
            raw_split_sizes = [grid[0] * grid[1] * grid[2] for grid in grids]
            local_cacheable = sum(raw_split_sizes) == int(pixel_values_videos.shape[0])

        globally_cacheable = self._distributed_flag(
            local_cacheable,
            pixel_values_videos.device,
            torch.distributed.ReduceOp.MIN,
        )
        if not globally_cacheable:
            self.bypassed_calls += 1
            logger.warning(
                "Validation video feature cache bypassed collectively: local_cacheable=%s pixel_rows=%s grid_rows=%s",
                local_cacheable,
                int(pixel_values_videos.shape[0]),
                sum(raw_split_sizes),
            )
            return encode_fn(pixel_values_videos, video_grid_thw)

        merged_split_sizes = [size // (spatial_merge_size**2) for size in raw_split_sizes]
        resolved_keys = [(str(cache_key), grid) for cache_key, grid in zip(cache_keys, grids)]
        pixel_chunks = torch.split(pixel_values_videos, raw_split_sizes, dim=0)

        available = {}
        missing = OrderedDict()
        call_hits = 0
        for index, key in enumerate(resolved_keys):
            entry = self.entries.get(key)
            if entry is not None:
                self.entries.move_to_end(key)
                available[key] = entry
                call_hits += 1
            elif key not in missing:
                missing[key] = index

        any_rank_missing = self._distributed_flag(
            bool(missing),
            pixel_values_videos.device,
            torch.distributed.ReduceOp.MAX,
        )
        if any_rank_missing and missing:
            miss_indices = list(missing.values())
            miss_pixels = torch.cat([pixel_chunks[index] for index in miss_indices], dim=0)
            miss_grids = torch.stack([video_grid_thw[index] for index in miss_indices], dim=0)
            fresh_main, fresh_deepstack = encode_fn(miss_pixels, miss_grids)
            if len(fresh_main) != len(miss_indices):
                raise RuntimeError("validation video feature encoder returned an unexpected media count")
            fresh_deepstack_splits = [
                torch.split(
                    layer,
                    [merged_split_sizes[index] for index in miss_indices],
                    dim=0,
                )
                for layer in fresh_deepstack
            ]
            for fresh_index, (key, _) in enumerate(missing.items()):
                # Clone split views so each cached media item does not retain a
                # whole miss-batch backing allocation.
                entry = (
                    fresh_main[fresh_index].detach().clone(),
                    tuple(layer_splits[fresh_index].detach().clone() for layer_splits in fresh_deepstack_splits),
                )
                available[key] = entry
                self.entries[key] = entry
                self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
        elif any_rank_missing:
            # FSDP vision layers execute collectives.  If another rank has a
            # miss, a fully cached rank must still enter the native encoder
            # exactly once to keep collective counts aligned.  Its first local
            # media item is a deterministic dummy and the result is discarded.
            encode_fn(pixel_chunks[0], video_grid_thw[:1])
            self.sync_dummy_encodes += 1
        else:
            self.global_all_hit_calls += 1

        ordered_entries = [available[key] for key in resolved_keys]
        main = tuple(entry[0] for entry in ordered_entries)
        deepstack_layer_count = len(ordered_entries[0][1]) if ordered_entries else 0
        deepstack = [
            torch.cat([entry[1][layer] for entry in ordered_entries], dim=0) for layer in range(deepstack_layer_count)
        ]

        call_misses = len(missing)
        self.calls += 1
        self.hits += call_hits
        self.misses += call_misses
        logger.info(
            "Validation video feature cache: call=%s requested=%s hits=%s "
            "unique_misses=%s entries=%s/%s cumulative_hits=%s cumulative_misses=%s "
            "sync_dummy_encodes=%s global_all_hit_calls=%s",
            self.calls,
            len(resolved_keys),
            call_hits,
            call_misses,
            len(self.entries),
            self.capacity,
            self.hits,
            self.misses,
            self.sync_dummy_encodes,
            self.global_all_hit_calls,
        )
        return main, deepstack

    def clear(self):
        stats = {
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "sync_dummy_encodes": self.sync_dummy_encodes,
            "global_all_hit_calls": self.global_all_hit_calls,
            "bypassed_calls": self.bypassed_calls,
        }
        self.entries.clear()
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.sync_dummy_encodes = 0
        self.global_all_hit_calls = 0
        self.bypassed_calls = 0
        return stats


class HFModelMethods:
    def _configure_validation_video_feature_cache(self):
        self._cosmos_validation_video_cache_keys = None
        self._cosmos_validation_video_cache_active = False
        self._cosmos_validation_video_feature_cache = None
        self._cosmos_validation_video_feature_target = None
        raw_capacity = os.environ.get("COSMOS_VALIDATION_VIDEO_FEATURE_CACHE_SIZE", "0")
        try:
            capacity = int(raw_capacity)
        except ValueError as exc:
            raise ValueError("COSMOS_VALIDATION_VIDEO_FEATURE_CACHE_SIZE must be an integer") from exc
        if capacity < 0:
            raise ValueError("COSMOS_VALIDATION_VIDEO_FEATURE_CACHE_SIZE must be non-negative")
        if capacity == 0:
            return
        if getattr(self.hf_config, "model_type", None) != "qwen3_vl":
            logger.warning(
                "Validation video feature cache requested for unsupported model type %s; disabled",
                getattr(self.hf_config, "model_type", None),
            )
            return

        target = getattr(self.model, "model", None)
        # Cosmos-RL's repository-owned Qwen3-VL compatibility forward merges
        # image/video inputs and calls ``get_image_features`` directly.  Hook
        # that common native encoder entry point; wrapping
        # ``get_video_features`` would be bypassed by the active runtime patch.
        original = getattr(target, "get_image_features", None)
        visual = getattr(target, "visual", None)
        spatial_merge_size = getattr(visual, "spatial_merge_size", None)
        if target is None or not callable(original) or not spatial_merge_size:
            raise RuntimeError("Qwen3-VL validation video feature cache could not resolve the native vision encoder")

        cache = _ValidationVideoFeatureCache(capacity)

        def cached_get_image_features(
            target_self,
            pixel_values: torch.Tensor,
            grid_thw: Optional[torch.Tensor] = None,
        ):
            del target_self
            if not self._cosmos_validation_video_cache_active:
                return original(pixel_values, grid_thw)
            return cache.get_or_encode(
                self._cosmos_validation_video_cache_keys,
                pixel_values,
                grid_thw,
                original,
                int(spatial_merge_size),
            )

        target.get_image_features = MethodType(cached_get_image_features, target)
        self._cosmos_validation_video_feature_cache = cache
        self._cosmos_validation_video_feature_target = target
        logger.info(
            "Enabled on-demand validation video feature cache: model_type=qwen3_vl "
            "capacity=%s device_resident=true training_enabled=false disk_cache=false",
            capacity,
        )

    def clear_validation_video_feature_cache(self):
        cache = self._cosmos_validation_video_feature_cache
        if cache is None:
            return None
        stats = cache.clear()
        logger.info(
            "Cleared validation video feature cache: calls=%s hits=%s misses=%s "
            "entries=%s bypassed_calls=%s sync_dummy_encodes=%s "
            "global_all_hit_calls=%s",
            stats["calls"],
            stats["hits"],
            stats["misses"],
            stats["entries"],
            stats["bypassed_calls"],
            stats["sync_dummy_encodes"],
            stats["global_all_hit_calls"],
        )
        return stats

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        cosmos_video_cache_keys = kwargs.pop("cosmos_video_cache_keys", None)
        kwargs_filtered = {k: v for k, v in kwargs.items() if k in self.model_forward_valid_kwargs}

        if "valid_input_len" in kwargs:
            kwargs_filtered["valid_input_len"] = kwargs["valid_input_len"]

        cache = self._cosmos_validation_video_feature_cache
        cache_active = cache is not None and not self.training and not torch.is_grad_enabled()
        if cache is not None and self.training and cache.entries:
            # A pre-train validation phase must never retain feature tensors
            # into gradient-enabled training.
            self.clear_validation_video_feature_cache()
        self._cosmos_validation_video_cache_active = cache_active
        self._cosmos_validation_video_cache_keys = cosmos_video_cache_keys
        try:
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                *args,
                **kwargs_filtered,
            )
        finally:
            self._cosmos_validation_video_cache_keys = None
            self._cosmos_validation_video_cache_active = False
        return out
