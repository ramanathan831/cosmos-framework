# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone, structured video annotation configuration (no runtime toolkit)."""

from dataclasses import dataclass, field


@dataclass
class VideoReasoningAnnotationGeminiConfig:
    api_key: str = ""
    model: str = ""
    media_resolution: str = "MEDIA_RESOLUTION_LOW"
    temperature: float = 0.3
    max_output_tokens: int = 8192
    timeout: int = 120


@dataclass
class VideoReasoningAnnotationOpenAIConfig:
    api_key: str = ""
    base_url: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class VideoReasoningAnnotationLLMConfig:
    backend: str = "gemini"
    gemini: VideoReasoningAnnotationGeminiConfig = field(default_factory=VideoReasoningAnnotationGeminiConfig)
    openai: VideoReasoningAnnotationOpenAIConfig = field(default_factory=VideoReasoningAnnotationOpenAIConfig)


@dataclass
class VideoReasoningAnnotationWorkflowConfig:
    steps: list[str] = field(default_factory=lambda: ["0", "1a", "1b", "1c", "2", "3", "4"])
    mode: str = "auto"
    max_workers: int = 4
    max_video_length_sec: int = 300
    chunk_duration_options: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 30])
    max_chunks: int = 10
    highlight_before_sec: float = 3.0
    highlight_after_sec: float = 3.0
    long_video_threshold_sec: int = 60
    long_video_sample_fps: float = 0.5
    long_video_max_frames: int = 60
    qa_types: list[str] = field(
        default_factory=lambda: [
            "mcq",
            "bcq",
            "open_qa",
            "causal_linkage",
            "temporal_localization",
            "temporal_event_desc",
            "scene_description",
            "event_summary",
        ]
    )


@dataclass
class VideoReasoningAnnotationDataConfig:
    video_root: str = ""
    input_jsonl_files: list[str] = field(default_factory=list)
    filter_field: str | None = None


@dataclass
class VideoReasoningAnnotationConfig:
    vlm: VideoReasoningAnnotationLLMConfig = field(default_factory=VideoReasoningAnnotationLLMConfig)
    llm: VideoReasoningAnnotationLLMConfig = field(default_factory=VideoReasoningAnnotationLLMConfig)
    workflow: VideoReasoningAnnotationWorkflowConfig = field(default_factory=VideoReasoningAnnotationWorkflowConfig)
    data: VideoReasoningAnnotationDataConfig = field(default_factory=VideoReasoningAnnotationDataConfig)
    license: str = ""
    description_extra: str = ""
    prompts_module: str = ""


@dataclass
class AnnotationConfig:
    results_dir: str = ""
    video_reasoning_annotation: VideoReasoningAnnotationConfig = field(default_factory=VideoReasoningAnnotationConfig)


def validate_config(config: AnnotationConfig) -> None:
    """Reject misspelled steps and invalid resource limits before any API calls."""
    workflow = config.video_reasoning_annotation.workflow
    if not config.results_dir:
        raise ValueError("results_dir is required")
    if not workflow.steps or set(workflow.steps) - {"0", "1a", "1b", "1c", "2", "3", "4"}:
        raise ValueError("steps must select from 0, 1a, 1b, 1c, 2, 3, 4")
    if workflow.mode not in {"auto", "anomaly", "normal"}:
        raise ValueError("mode must be auto, anomaly, or normal")
    for name in (
        "max_workers",
        "max_video_length_sec",
        "max_chunks",
        "long_video_threshold_sec",
        "long_video_sample_fps",
        "long_video_max_frames",
    ):
        if getattr(workflow, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not workflow.chunk_duration_options or any(v <= 0 for v in workflow.chunk_duration_options):
        raise ValueError("chunk_duration_options must contain positive durations")
    if min(workflow.highlight_before_sec, workflow.highlight_after_sec) < 0:
        raise ValueError("highlight durations must be nonnegative")
