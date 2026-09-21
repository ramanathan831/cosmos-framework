# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the resumable video annotation pipeline from a YAML configuration."""

import argparse
import logging


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration; credentials come from the environment")
    parser.add_argument("--print-defaults", action="store_true", help="Print the structured YAML template and exit")
    parser.add_argument("--results-dir", help="Override the configuration output directory")
    args = parser.parse_args(argv)

    from omegaconf import OmegaConf

    from cosmos_framework.inference.video_annotation.config import AnnotationConfig, validate_config
    from cosmos_framework.inference.video_annotation.inference import run_video_reasoning_annotation_inference

    if args.print_defaults:
        print(OmegaConf.to_yaml(OmegaConf.structured(AnnotationConfig)))
        return
    if not args.config:
        parser.error("--config is required unless --print-defaults is selected")
    config = OmegaConf.merge(OmegaConf.structured(AnnotationConfig), OmegaConf.load(args.config))
    if args.results_dir:
        config.results_dir = args.results_dir
    # Convert to dataclasses so workflow parameters can be forwarded to clients
    # without mutating an OmegaConf struct or writing credentials to disk.
    config = OmegaConf.to_object(config)
    validate_config(config)
    logging.basicConfig(level=logging.INFO)
    run_video_reasoning_annotation_inference(config, config.results_dir)


if __name__ == "__main__":
    main()
