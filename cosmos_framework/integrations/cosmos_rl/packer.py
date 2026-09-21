# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos video identity and collation extensions for native Cosmos-RL."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import qwen_vl_utils.vision_process as vision_process
import torch
from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import (
    IGNORE_LABEL_ID,
    HFVLMDataPacker,
    extract_vision_info,
    fetch_video_frames,
    retrieve_not_none_values,
)
from cosmos_rl.utils.logging import logger
from PIL import Image
from qwen_vl_utils import fetch_image

from cosmos_framework.inference.reasoner.video_pixel_bounds import normalize_video_pixel_bounds


def qwen_vl_process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    return_video_kwargs: bool = False,
    return_video_metadata: bool = False,
    image_patch_size: int = 14,
) -> Tuple[
    Optional[List[Image.Image]],
    Optional[List[Union[torch.Tensor, List[Image.Image]]]],
    Optional[Dict[str, Any]],
]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, image_patch_size=image_patch_size))
        elif "video" in vision_info:
            normalize_video_pixel_bounds(
                vision_info,
                image_patch_size,
                vision_process,
            )
            # Resolve fetch_video dynamically from vision_process.  Spawned
            # DataLoader workers install the PyNv processed-video cache on
            # this module before/while selecting the forced backend.  A
            # function imported by value here would retain Qwen's original
            # uncached function and silently bypass that lazy in-training
            # cache for every sample.
            video_input, video_sample_fps = vision_process.fetch_video(
                vision_info,
                return_video_sample_fps=True,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        elif "frame_dir" in vision_info:
            video_input, video_sample_fps = fetch_video_frames(
                vision_info,
                image_patch_size=image_patch_size,
                return_video_metadata=return_video_metadata,
            )
            video_sample_fps_list.append(video_sample_fps)
            video_inputs.append(video_input)
        else:
            raise ValueError("image, image_url, frame_dir or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None

    video_kwargs = {"do_sample_frames": False}
    if not return_video_metadata:  # BC for qwen2.5vl
        video_kwargs.update({"fps": video_sample_fps_list})

    if return_video_kwargs:
        return image_inputs, video_inputs, video_kwargs
    return image_inputs, video_inputs


class HFVLMDataPackerMethods:
    def _collate_fn(self, processed_samples: List[Dict[str, Any]], computed_max_len: int) -> Dict[str, Any]:
        cosmos_video_cache_keys = []
        cosmos_video_cache_keys_valid = True
        for sample in processed_samples:
            sample_grid = sample.get("video_grid_thw")
            if sample_grid is None:
                continue
            sample_keys = sample.get("cosmos_video_cache_keys")
            grid_rows = int(sample_grid.shape[0])
            if not isinstance(sample_keys, (list, tuple)) or len(sample_keys) != grid_rows:
                cosmos_video_cache_keys_valid = False
                break
            cosmos_video_cache_keys.extend(str(key) for key in sample_keys)

        pixel_values_videos = [x["pixel_values_videos"] for x in processed_samples]
        video_grid_thw = [x["video_grid_thw"] for x in processed_samples]
        second_per_grid_ts = [x["second_per_grid_ts"] for x in processed_samples]
        pixel_values = [x["pixel_values"] for x in processed_samples]
        image_grid_thw = [x["image_grid_thw"] for x in processed_samples]
        pixel_values_videos_lengths_per_sample = [
            x["pixel_values_videos_lengths_per_sample"] for x in processed_samples
        ]
        pixel_values_lengths_per_sample = [x["pixel_values_lengths_per_sample"] for x in processed_samples]
        aspect_ratio_ids = [x["aspect_ratio_ids"] for x in processed_samples]
        aspect_ratio_mask = [x["aspect_ratio_mask"] for x in processed_samples]
        image_sizes = [x["image_sizes"] for x in processed_samples]
        batch_num_images = [x["batch_num_images"] for x in processed_samples]

        pixel_values_videos = retrieve_not_none_values(pixel_values_videos)

        pixel_values_videos_lengths_per_sample = [x for x in pixel_values_videos_lengths_per_sample if x is not None]
        pixel_values_videos_lengths_per_sample = (
            pixel_values_videos_lengths_per_sample if len(pixel_values_videos_lengths_per_sample) > 0 else None
        )

        video_grid_thw = retrieve_not_none_values(video_grid_thw)

        second_per_grid_ts = retrieve_not_none_values(second_per_grid_ts)

        pixel_values = retrieve_not_none_values(pixel_values)

        pixel_values_lengths_per_sample = [x for x in pixel_values_lengths_per_sample if x is not None]
        pixel_values_lengths_per_sample = (
            pixel_values_lengths_per_sample if len(pixel_values_lengths_per_sample) > 0 else None
        )

        image_grid_thw = retrieve_not_none_values(image_grid_thw)

        aspect_ratio_ids = retrieve_not_none_values(aspect_ratio_ids)

        aspect_ratio_mask = retrieve_not_none_values(aspect_ratio_mask)

        image_sizes = retrieve_not_none_values(image_sizes)

        batch_num_images = retrieve_not_none_values(batch_num_images)

        # Shape description:
        #
        # pixel_values_[videos/images]: (BATCH_SIZE, N_PATCH, HIDDEN_SIZE)
        # [video/image]_grid_thw: (BATCH_SIZE, 3)
        # second_per_grid_ts: (BATCH_SIZE, 1)
        # pixel_values_[videos/images]_lengths_per_sample: (BATCH_SIZE, 1)
        batch = {}
        if pixel_values_videos is not None:
            batch["pixel_values_videos"] = pixel_values_videos

        if video_grid_thw is not None:
            batch["video_grid_thw"] = video_grid_thw

        if (
            cosmos_video_cache_keys_valid
            and cosmos_video_cache_keys
            and video_grid_thw is not None
            and len(cosmos_video_cache_keys) == int(video_grid_thw.shape[0])
        ):
            # This metadata is consumed by HFModel before its kwargs are
            # filtered against the Hugging Face forward signature.  Keeping it
            # as Python strings avoids tensor transfers and keeps checkpoint
            # state untouched.
            batch["cosmos_video_cache_keys"] = cosmos_video_cache_keys

        if second_per_grid_ts is not None:
            batch["second_per_grid_ts"] = second_per_grid_ts

        if pixel_values_videos_lengths_per_sample is not None:
            batch["pixel_values_videos_lengths_per_sample"] = torch.tensor(
                pixel_values_videos_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        if pixel_values is not None:
            batch["pixel_values"] = pixel_values

        if image_grid_thw is not None:
            batch["image_grid_thw"] = image_grid_thw

        if pixel_values_lengths_per_sample is not None:
            batch["pixel_values_lengths_per_sample"] = torch.tensor(
                pixel_values_lengths_per_sample, dtype=torch.long
            ).view(-1, 1)

        if aspect_ratio_ids is not None:
            batch["aspect_ratio_ids"] = aspect_ratio_ids

        if aspect_ratio_mask is not None:
            batch["aspect_ratio_mask"] = aspect_ratio_mask

        if image_sizes is not None:
            batch["image_sizes"] = image_sizes

        if batch_num_images is not None:
            batch["batch_num_images"] = batch_num_images

        # Pad input_ids and build the mask from the unpadded lengths.  Do not
        # infer padding from the token value: some tokenizers reuse a regular
        # vocabulary token as ``pad_token_id``.  An explicit attention mask is
        # also required by recent Transformers releases for multimodal models
        # whose MRoPE position ids are not monotonically increasing.  Without
        # it, Transformers can mis-detect an ordinary padded batch as packed
        # sequences and sever text-to-vision attention during SFT.
        batch["input_ids"] = torch.tensor(
            [
                x["input_ids"][:computed_max_len]
                + [self.tokenizer.pad_token_id] * (max(0, computed_max_len - len(x["input_ids"])))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        batch["attention_mask"] = torch.tensor(
            [
                [1] * min(len(x["input_ids"]), computed_max_len) + [0] * max(0, computed_max_len - len(x["input_ids"]))
                for x in processed_samples
            ],
            dtype=torch.long,
        )
        if "mm_token_type_ids" in processed_samples[0]:

            def _to_padded_mm_ids(x):
                ids = x.get("mm_token_type_ids")
                if ids is None:
                    ids = []
                elif isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                # Flatten 2D arrays (e.g., shape (1, seq_len)) to 1D
                if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
                    ids = [item for sublist in ids for item in sublist] if len(ids) > 1 else list(ids[0])
                truncated = ids[:computed_max_len]
                pad_len = computed_max_len - len(truncated)
                return truncated + [0] * max(0, pad_len)

            batch["mm_token_type_ids"] = torch.tensor(
                [_to_padded_mm_ids(x) for x in processed_samples],
                dtype=torch.long,
            )

        if "label_ids" in processed_samples[0]:
            batch["label_ids"] = torch.tensor(
                [
                    x["label_ids"][:computed_max_len]
                    + [IGNORE_LABEL_ID] * (max(0, computed_max_len - len(x["label_ids"])))
                    for x in processed_samples
                ],
                dtype=torch.long,
            )

        batch["logprob_masks"] = torch.tensor(
            [
                x["logprob_masks"][:computed_max_len] + [0] * (max(0, computed_max_len - len(x["logprob_masks"])))
                for x in processed_samples
            ],
            dtype=torch.bool,
        )

        assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["logprob_masks"].shape, (
            "The shapes of input_ids, attention_mask, and logprob_masks should be the same"
        )

        return batch

    @staticmethod
    def _extract_video_cache_keys(sample: "HFVLMDataPacker.Payload"):
        """Return stable video identities in processor traversal order.

        Only ordinary string paths are cacheable.  URLs, in-memory videos,
        frame lists, or malformed conversations deliberately return ``None``
        so the model takes its native uncached path.
        """
        messages = sample.get("messages") if isinstance(sample, dict) else sample
        if not isinstance(messages, list):
            return None

        keys = []
        for message in messages:
            if not isinstance(message, dict) and hasattr(message, "model_dump"):
                message = message.model_dump()
            if not isinstance(message, dict):
                return None
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                if "video" not in item and item.get("type") != "video":
                    continue
                value = item.get("video")
                if not isinstance(value, str) or "://" in value:
                    return None
                keys.append(os.path.realpath(os.path.expanduser(value)))
        return keys or None

    def sft_process_sample(self, sample: "HFVLMDataPacker.Payload") -> Dict[str, Any]:
        """
        Accepts either raw text or conversation format.
        """
        cosmos_video_cache_keys = self._extract_video_cache_keys(sample)
        result = self.get_policy_input(sample, add_generation_prompt=False)

        video_grid_thw = result.get("video_grid_thw")
        if (
            cosmos_video_cache_keys is not None
            and video_grid_thw is not None
            and len(cosmos_video_cache_keys) == int(video_grid_thw.shape[0])
        ):
            result["cosmos_video_cache_keys"] = cosmos_video_cache_keys

        max_len = getattr(self.config.policy, "model_max_length", None)
        if max_len is not None and len(result["input_ids"]) > max_len:
            media_type = self._detect_media_types(sample)
            has_vision = result.get("pixel_values") is not None or result.get("pixel_values_videos") is not None
            if has_vision:
                # Truncation is safe only if every vision placeholder token
                # falls within the kept prefix (indices 0..max_len-1).  If so,
                # only trailing text tokens are removed and pixel tensor
                # alignment is preserved.
                input_ids = result["input_ids"]
                vision_ids = {v for v in self.vision_ids if v is not None}
                last_vision_pos = -1
                for i, tok in enumerate(input_ids):
                    if tok in vision_ids:
                        last_vision_pos = i
                        if last_vision_pos >= max_len:
                            break
                if last_vision_pos >= max_len:
                    raise ValueError(
                        f"[{media_type}] Sample exceeds model_max_length after tokenization "
                        f"({len(result['input_ids'])} > {max_len}) and truncation would "
                        f"break vision token alignment, skipping."
                    )
            orig_len = len(result["input_ids"])
            result["input_ids"] = result["input_ids"][:max_len]
            if "label_ids" in result:
                result["label_ids"] = result["label_ids"][:max_len]
            if "logprob_masks" in result:
                result["logprob_masks"] = result["logprob_masks"][:max_len]
            if has_vision:
                logger.warning(
                    f"[{media_type}] Truncated vision sample from {orig_len} to "
                    f"{max_len} tokens (trailing text only, vision tokens preserved)."
                )
            else:
                logger.warning(f"[{media_type}] Truncated sample from {orig_len} to {max_len} tokens.")

        return result
