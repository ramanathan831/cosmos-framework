# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-aware SFT hook using the Framework-owned conversation dataset."""

import argparse
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Literal

import cosmos_rl.launcher.worker_entry
import cosmos_rl.policy.config
import pydantic
import toml
import torch.utils.data
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker
from cosmos_rl.utils.logging import logger

from cosmos_framework.data.reasoner.qa_dataset import (
    ReasoningConversationDataset,
    ReasoningDataPackerMixin,
)
from cosmos_framework.integrations.cosmos_rl.compat import install_runtime_extensions
from cosmos_framework.integrations.cosmos_rl.lifecycle_status import (
    append_terminal_status,
    is_lifecycle_status_owner,
)
from cosmos_framework.integrations.cosmos_rl.status_hooks import CosmosStatusLogger
from cosmos_framework.utils.workflow_status import Status, StatusLogger, Verbosity, set_status_logger

install_runtime_extensions()


ResponseMode = Literal["think", "answer", "hybrid"]


class VisionConfig(pydantic.BaseModel):
    """Vision processing options forwarded to the Cosmos-RL VLM packer."""

    resized_height: int | None = None
    resized_width: int | None = None
    min_pixels: int | None = None
    max_pixels: int | None = None
    total_pixels: int | None = None
    video_start: float | None = None
    video_end: float | None = None
    nframes: int | None = None
    fps: float | None = None
    min_frames: int | None = None
    max_frames: int | None = None


class CustomDatasetConfig(pydantic.BaseModel):
    annotation_path: str | list[str] = pydantic.Field()
    """One or more cosmos-video-reasoning-v1.0 annotation JSON files."""

    media_root: str | list[str] | None = None
    """Optional media root override. Defaults to each annotation file directory."""

    response_mode: ResponseMode = "answer"
    """Use answer-only, think+answer, or both forms for each item."""

    max_samples: int | None = None
    """Maximum number of samples exposed to Cosmos-RL after response expansion."""

    sample_stride: int = 1
    """Take every Nth raw item when subsampling."""

    sample_offset: int = 0
    """Start offset for raw-item subsampling."""

    @pydantic.model_validator(mode="after")
    def check_subsample_config(self):
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        return self


class CustomConfig(pydantic.BaseModel):
    train_dataset: CustomDatasetConfig = pydantic.Field()
    val_dataset: CustomDatasetConfig | None = None
    system_prompt: str = ""
    video_decoder: Literal["pynvvideocodec", "cpu", "torchvision"] = "pynvvideocodec"
    video_cache_size: int = pydantic.Field(default=2, ge=0)
    video_override_map: str | None = None
    vision: VisionConfig = pydantic.Field(
        default=VisionConfig(
            fps=1,
            max_pixels=81920,
        )
    )


class ReasoningHFVLMDataPacker(ReasoningDataPackerMixin, HFVLMDataPacker):
    """HF VLM packer with the task-aware Cosmos-Reason chat-template override."""


class CustomDataset(torch.utils.data.Dataset):
    """Torch dataset wrapper around the task-aware cosmos-video-reasoning Cosmos-RL dataset."""

    def __init__(
        self,
        config: cosmos_rl.policy.config.Config,
        custom_config: CustomConfig,
        dataset_config: CustomDatasetConfig,
    ):
        logger.info(
            "Creating task-aware dataset from annotation_path=%s, media_root=%s, "
            "response_mode=%s, max_samples=%s, sample_stride=%s, sample_offset=%s",
            dataset_config.annotation_path,
            dataset_config.media_root,
            dataset_config.response_mode,
            dataset_config.max_samples,
            dataset_config.sample_stride,
            dataset_config.sample_offset,
        )
        self.dataset = ReasoningConversationDataset(
            annotation_paths=dataset_config.annotation_path,
            media_roots=dataset_config.media_root,
            system_prompt=custom_config.system_prompt,
            vision_kwargs=custom_config.vision.model_dump(exclude_none=True),
            response_mode=dataset_config.response_mode,
        )
        self.config = config
        self.custom_config = custom_config
        self.dataset_config = dataset_config
        self._raw_length = int(getattr(self.dataset, "_raw_length", len(self.dataset)))
        self._length = self._compute_length()
        logger.info(
            "task-aware dataset raw_length=%s, exposed_length=%s",
            self._raw_length,
            self._length,
        )

    def setup(self, config):
        """Setup method required by the SFT trainer."""

    def _raw_subsample_length(self) -> int:
        if self.dataset_config.sample_offset >= self._raw_length:
            return 0
        return ((self._raw_length - self.dataset_config.sample_offset - 1) // self.dataset_config.sample_stride) + 1

    def _compute_length(self) -> int:
        if self.dataset_config.response_mode == "hybrid":
            length = self._raw_subsample_length() * 2
        else:
            length = self._raw_subsample_length()
        if self.dataset_config.max_samples is not None:
            length = min(length, self.dataset_config.max_samples)
        return length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> list[dict]:
        if not 0 <= index < self._length:
            raise IndexError(index)

        if self.dataset_config.response_mode == "hybrid":
            raw_index = self.dataset_config.sample_offset + (index // 2) * self.dataset_config.sample_stride
            dataset_index = raw_index if index % 2 == 0 else self._raw_length + raw_index
        else:
            dataset_index = self.dataset_config.sample_offset + index * self.dataset_config.sample_stride

        return self.dataset[dataset_index]


def _get_results_dir() -> str | None:
    job_id = os.environ.get("COSMOS_API_JOB_ID")
    results_base = os.environ.get("COSMOS_API_RESULTS_DIR")
    if job_id and results_base:
        return os.path.join(results_base, job_id)
    return None


def _is_master_rank() -> bool:
    """Check if this entrypoint process owns terminal lifecycle logging."""
    return is_lifecycle_status_owner()


def _write_fallback_status(status_file: str, status: str, message: str) -> None:
    entries: list[dict[str, Any]] = []
    if os.path.exists(status_file):
        with open(status_file, encoding="utf-8") as f:
            try:
                content = json.load(f)
                entries = content if isinstance(content, list) else [content]
            except json.JSONDecodeError:
                entries = []

    entries.append(
        {
            "date": datetime.now().isoformat(),
            "status": status,
            "message": message,
            "data": {"component": "Cosmos-RL task-aware SFT"},
        }
    )
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def monitor_status(name: str = "Cosmos-RL", mode: str = "sft"):
    """Log STARTED/RUNNING/FAILURE lifecycle status around this entrypoint."""

    def inner(runner):
        @wraps(runner)
        def wrapper(*args, **kwargs):
            status_file = None
            status_logger = None
            if _is_master_rank():
                results_dir = _get_results_dir()
                if results_dir is not None:
                    os.makedirs(results_dir, exist_ok=True)
                    status_file = os.path.join(results_dir, "status.json")

                if status_file is not None and True:
                    status_logger = StatusLogger(
                        filename=status_file,
                        is_master=True,
                        verbosity=Verbosity.INFO,
                        append=True,
                    )
                    set_status_logger(status_logger)
                    status_logger.write(
                        status_level=Status.STARTED,
                        message=f"Starting {name} {mode}",
                    )
                elif status_file is not None:
                    _write_fallback_status(
                        status_file,
                        "STARTED",
                        f"Starting {name} {mode}",
                    )

                logger.info("Lifecycle status will be logged to %s", status_file)

            try:
                result = runner(*args, **kwargs)
                if status_file is not None:
                    append_terminal_status(
                        status_file,
                        "SUCCESS",
                        f"{name} {mode} completed successfully",
                    )
                return result
            except (KeyboardInterrupt, SystemExit) as exc:
                if status_file is not None:
                    append_terminal_status(
                        status_file,
                        "FAILURE",
                        f"{name} {mode} interrupted: {exc}",
                    )
                raise
            except Exception as exc:
                if status_file is not None:
                    append_terminal_status(
                        status_file,
                        "FAILURE",
                        f"{name} {mode} failed: {exc}",
                    )
                raise

        return wrapper

    return inner


def _cosmos_logging_enabled(config: cosmos_rl.policy.config.Config) -> bool:
    loggers = config.logging.logger if hasattr(config.logging, "logger") else []
    return "workflow_status" in loggers if isinstance(loggers, list) else loggers == "workflow_status"


def _build_status_hooks(config: cosmos_rl.policy.config.Config):
    if not _cosmos_logging_enabled(config):
        return [], {}

    if not os.environ.get("COSMOS_API_JOB_ID"):
        logger.info("Cosmos logger requested but COSMOS_API_JOB_ID is not set")
        return [], {}

    cosmos_logger = CosmosStatusLogger(experiment_name=config.logging.experiment_name or "Cosmos-RL task-aware SFT")
    logger.info(
        "Cosmos status updates will be logged to %s",
        cosmos_logger._get_status_file_path(),
    )
    return [cosmos_logger.log_status], cosmos_logger.get_hooks()


def get_data_packer(config: cosmos_rl.policy.config.Config) -> BaseDataPacker:
    return ReasoningHFVLMDataPacker()


def configure_video_decoder(custom_config: CustomConfig) -> dict[str, object] | None:
    """Register the decoder selected by the task-aware runtime contract."""
    from cosmos_framework.integrations.cosmos_rl.compat import configure_patch_embedding

    configure_patch_embedding()
    if os.environ.get("COSMOS_ROLE") == "Controller":
        return None
    if custom_config.video_decoder == "pynvvideocodec":
        os.environ.pop("COSMOS_DATALOADER_VIDEO_DECODER", None)
        from cosmos_framework.inference.reasoner.pynv_video_reader import register_pynv_video_reader

        decoder_info = register_pynv_video_reader(
            cache_size=custom_config.video_cache_size,
            video_override_map=custom_config.video_override_map,
        )
        logger.info("GPU video decoder registered: %s", decoder_info)
        return decoder_info
    if custom_config.video_decoder in ("cpu", "torchvision"):
        os.environ["COSMOS_DATALOADER_VIDEO_DECODER"] = "system_pyav"
        from cosmos_framework.inference.reasoner.system_pyav_video_reader import (
            register_system_pyav_video_reader,
        )

        register_system_pyav_video_reader()
        decoder_info = {
            "backend": "torchvision",
            "implementation": "system_pyav_sparse",
            "requested": custom_config.video_decoder,
        }
        logger.info("System PyAV video decoder registered: %s", decoder_info)
        return decoder_info
    raise ValueError("custom.video_decoder must be pynvvideocodec, torchvision, or cpu")


@monitor_status(name="Cosmos-RL Cosmos VL Reason task-aware", mode="sft")
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True, help="Path to config file.")
    args = parser.parse_known_args()[0]

    with open(args.config, encoding="utf-8") as f:
        config_kwargs = toml.load(f)
    config = cosmos_rl.policy.config.Config.from_dict(config_kwargs)
    custom_config = CustomConfig.model_validate(config_kwargs.get("custom", {}))
    configure_video_decoder(custom_config)

    if os.environ.get("COSMOS_ROLE") == "Controller":
        output_dir = Path(config.train.output_dir).resolve().parent
        output_dir.mkdir(parents=True, exist_ok=True)
        config_kwargs_to_save = config.model_dump()
        config_kwargs_to_save["custom"] = custom_config.model_dump()
        config_path = output_dir / "config.toml"
        config_path.write_text(toml.dumps(config_kwargs_to_save))
        logger.info("Saved config to %s", config_path)

    def get_train_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset:
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))
        return CustomDataset(
            config=config,
            custom_config=custom_cfg,
            dataset_config=custom_cfg.train_dataset,
        )

    def get_val_dataset(
        config: cosmos_rl.policy.config.Config,
    ) -> torch.utils.data.Dataset | None:
        custom_cfg = CustomConfig.model_validate(config.model_dump().get("custom", {}))
        if custom_cfg.val_dataset is None:
            logger.info("No validation dataset specified")
            return None
        return CustomDataset(
            config=config,
            custom_config=custom_cfg,
            dataset_config=custom_cfg.val_dataset,
        )

    custom_logger_fns, hook_fns = _build_status_hooks(config)

    cosmos_rl.launcher.worker_entry.main(
        dataset=get_train_dataset,
        val_dataset=get_val_dataset if custom_config.val_dataset else None,
        data_packer=get_data_packer,
        val_data_packer=get_data_packer,
        custom_logger_fns=custom_logger_fns,
        hook_fns=hook_fns,
    )


if __name__ == "__main__":
    main()
