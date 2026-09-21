# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "cosmos_framework/integrations/cosmos_rl/runtime_dependency_contract.py"
PACKER = Path(__file__).parents[2] / "cosmos_framework/integrations/cosmos_rl/packer.py"
SPEC = importlib.util.spec_from_file_location("runtime_dependency_contract", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
missing_deepep_symbols = MODULE.missing_deepep_symbols
repair_qwen_pynv_worker_source = MODULE.repair_qwen_pynv_worker_source
repair_vllm_conv3d_source = MODULE.repair_vllm_conv3d_source


OLD_SOURCE = """
from vllm.utils.torch_utils import is_torch_equal
if self.enable_linear and (is_torch_equal("2.9.0") or is_torch_equal("2.9.1")):
    return self._forward_mulmat(x)
"""

QWEN_SOURCE = """
FORCE_QWENVL_VIDEO_READER = os.getenv("FORCE_QWENVL_VIDEO_READER", None)


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_QWENVL_VIDEO_READER is not None:
        video_reader_backend = FORCE_QWENVL_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"qwen-vl-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, video_metadata, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

def fetch_video(ele: Dict[str, Any], image_patch_size: int = 14, return_video_sample_fps: bool = False,
                return_video_metadata: bool = False) -> Union[torch.Tensor, List[Image.Image]]:
    image_factor = image_patch_size * SPATIAL_MERGE_SIZE
"""


def test_repair_vllm_conv3d_source_is_idempotent() -> None:
    repaired, changed = repair_vllm_conv3d_source(OLD_SOURCE)
    assert changed
    assert "is_torch_equal_or_newer" in repaired
    assert 'is_torch_equal_or_newer("2.9.0")' in repaired
    assert repair_vllm_conv3d_source(repaired) == (repaired, False)


def test_repair_vllm_conv3d_rejects_unknown_source() -> None:
    with pytest.raises(RuntimeError, match="Unrecognized"):
        repair_vllm_conv3d_source("def forward_cuda(self, x): pass")


def test_repair_qwen_pynv_worker_source_is_idempotent() -> None:
    repaired, changed = repair_qwen_pynv_worker_source(QWEN_SOURCE)
    assert changed
    assert "def _ensure_forced_video_reader(" in repaired
    assert "_ensure_forced_video_reader(video_reader_backend)" in repaired
    assert 'os.getenv("COSMOS_PYNV_DECODER_CACHE_SIZE", "4")' in repaired
    assert "strict GPU video decoding failed; CPU fallback is disabled" in repaired
    assert "normalize_video_pixel_bounds(ele, image_patch_size" in repaired
    assert "sys.modules[__name__]" in repaired
    assert repair_qwen_pynv_worker_source(repaired) == (repaired, False)


def test_repair_qwen_pynv_worker_source_upgrades_existing_worker_contract() -> None:
    repaired, _ = repair_qwen_pynv_worker_source(QWEN_SOURCE)
    worker_only = repaired.replace(
        '    if os.getenv("FORCE_QWENVL_VIDEO_READER", FORCE_QWENVL_VIDEO_READER) == "pynvvideocodec":\n'
        "        from cosmos_framework.inference.reasoner.video_pixel_bounds import normalize_video_pixel_bounds\n\n"
        "        normalize_video_pixel_bounds(ele, image_patch_size, sys.modules[__name__])\n",
        "",
    )
    upgraded, changed = repair_qwen_pynv_worker_source(worker_only)
    assert changed
    assert "normalize_video_pixel_bounds(ele, image_patch_size" in upgraded
    assert repair_qwen_pynv_worker_source(upgraded) == (upgraded, False)


def test_repair_qwen_pynv_worker_source_rejects_unknown_source() -> None:
    with pytest.raises(RuntimeError, match="Unrecognized qwen-vl-utils"):
        repair_qwen_pynv_worker_source("def fetch_video(): pass")


def test_repair_qwen_pynv_worker_source_rejects_stale_worker_contract() -> None:
    repaired, _ = repair_qwen_pynv_worker_source(QWEN_SOURCE)
    stale = repaired.replace(
        '        decoder_cache_size=int(os.getenv("COSMOS_PYNV_DECODER_CACHE_SIZE", "4")),\n',
        "",
    )
    with pytest.raises(RuntimeError, match="Unrecognized qwen-vl-utils"):
        repair_qwen_pynv_worker_source(stale)


def test_deepep_symbol_contract_reports_only_missing_symbols() -> None:
    symbols = "\n".join(
        (
            "deep_ep::internode_ll::clean_mask_buffer(int*)",
            "deep_ep::internode_ll::update_mask_buffer(int*)",
        )
    )
    assert missing_deepep_symbols(symbols) == ["internode_ll::query_mask_buffer"]


def test_qwen_packer_normalizes_video_bounds_before_dynamic_cached_fetch() -> None:
    source = PACKER.read_text(encoding="utf-8")
    function = source.split("def qwen_vl_process_vision_info(", 1)[1].split("\ndef process_vision_info(", 1)[0]

    assert "normalize_video_pixel_bounds(" in function
    assert "vision_process.fetch_video(" in function
    assert function.index("normalize_video_pixel_bounds(") < function.index("vision_process.fetch_video(")
    assert "from qwen_vl_utils import fetch_image, fetch_video" not in source
