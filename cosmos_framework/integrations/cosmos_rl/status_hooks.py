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

"""Status hooks for the optional Cosmos-RL training backend."""

import os
from typing import Any, Callable, Dict, Optional

from cosmos_rl.utils.logging import logger


class CosmosStatusLogger:
    """Cosmos-compatible status logger that writes to status.json file.

    This class replicates the Cosmos status logging functionality, writing
    status updates to a JSON file in the format expected by Cosmos/NVAIE.

    The status file is written to:
        {COSMOS_API_RESULTS_DIR}/{COSMOS_API_JOB_ID}/status.json

    Usage:
        ```python
        cosmos_logger = CosmosStatusLogger(
            experiment_name="sft-training"
        )

        main(
            custom_logger_fns=[cosmos_logger.log_status],
            hook_fns=cosmos_logger.get_hooks(),
        )
        ```
    """

    def __init__(
        self,
        experiment_name: str = "cosmos-rl-experiment",
        status_file_path: Optional[str] = None,
    ):
        """Initialize Cosmos status logger.

        Args:
            experiment_name: Name of the experiment/component for logging.
            status_file_path: Optional explicit path to status.json file.
                If not provided, uses COSMOS_API_RESULTS_DIR/COSMOS_API_JOB_ID/status.json
        """
        self.experiment_name = experiment_name
        self._status_file_path = status_file_path
        self._status_logger = None

    def _get_status_file_path(self) -> Optional[str]:
        """Get the Cosmos status file path based on environment variables.

        Returns:
            Path to status.json file, or None if COSMOS_API_JOB_ID not set.
        """
        if self._status_file_path:
            return self._status_file_path
        if os.environ.get("COSMOS_STATUS_FILE"):
            return os.environ["COSMOS_STATUS_FILE"]

        job_id = os.environ.get("COSMOS_API_JOB_ID")
        if not job_id:
            logger.debug("COSMOS_API_JOB_ID not set, skipping status.json logging")
            return None

        results_base = os.environ.get("COSMOS_API_RESULTS_DIR")
        if not results_base:
            logger.warning("COSMOS_API_RESULTS_DIR is required when COSMOS_API_JOB_ID is set")
            return None
        results_dir = os.path.join(results_base, job_id)
        os.makedirs(results_dir, exist_ok=True)
        return os.path.join(results_dir, "status.json")

    def _get_status_logger(self):
        """Get or create the Cosmos StatusLogger instance."""
        if self._status_logger is None:
            status_file = self._get_status_file_path()
            if status_file is None:
                return None

            from cosmos_framework.utils.workflow_status import StatusLogger, Verbosity

            is_master = int(os.environ.get("NODE_RANK", 0)) == 0
            self._status_logger = StatusLogger(
                filename=status_file,
                is_master=is_master,
                verbosity=Verbosity.INFO,
                append=True,
            )

        return self._status_logger

    @staticmethod
    def _convert_tensors_to_scalars(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert any PyTorch tensors in a dict to Python scalars for JSON serialization."""
        result = {}
        for k, v in data.items():
            if hasattr(v, "item"):  # PyTorch tensor
                result[k] = v.item()
            elif hasattr(v, "tolist"):  # NumPy array
                result[k] = v.tolist()
            else:
                result[k] = v
        return result

    def log_status(self, report_data: Dict[str, Any], step: int) -> None:
        """Custom logger function for Cosmos status updates.

        This replaces the hardcoded log_cosmos_status calls with a pluggable
        custom logger function that writes to status.json.

        Args:
            report_data: Dictionary containing training/validation metrics.
            step: Current training step.
        """
        status_logger = self._get_status_logger()
        if status_logger is None:
            return

        try:
            from datetime import timedelta

            # Convert any tensors to Python scalars for JSON serialization
            report_data = self._convert_tensors_to_scalars(report_data)

            # Get epoch info from either training or validation report_data
            current_epoch = report_data.get("train/cur_epoch", report_data.get("val/cur_epoch"))
            max_epochs = report_data.get("train/total_epochs", report_data.get("val/train_epochs"))
            total_steps = report_data.get("train/total_steps", report_data.get("val/total_steps", 0))
            steps_per_epoch = report_data.get("steps_per_epoch", 1)

            # Use epoch-based logging if provided, otherwise fall back to step-based
            # But calculate ETA based on remaining STEPS for more accurate progress
            iteration_time = report_data.get("train/iteration_time", 1.0)

            if current_epoch is not None and max_epochs is not None:
                current_value = current_epoch
                max_value = max_epochs
                # ETA based on remaining steps (more accurate than remaining epochs)
                remaining_steps = max(total_steps - step, 0)
                eta_seconds = remaining_steps * iteration_time
                estimated_time_per_unit = iteration_time * steps_per_epoch
                log_key = "epoch"
            else:
                current_value = step
                max_value = total_steps if total_steps > 0 else step
                remaining_steps = max(max_value - current_value, 0)
                eta_seconds = remaining_steps * iteration_time
                estimated_time_per_unit = iteration_time
                log_key = "step"

            # Create Cosmos-compatible status data
            cosmos_data = {
                "component": self.experiment_name,
                log_key: current_value,
                f"max_{log_key}": max_value,
                "time_per_epoch": str(timedelta(seconds=estimated_time_per_unit)),
                "eta": str(timedelta(seconds=eta_seconds)),
            }

            # Create summary message based on available metrics
            if "checkpoint/event" in report_data:
                message = (
                    f"Checkpoint {report_data['checkpoint/event']}: "
                    f"{report_data.get('checkpoint/identifier', 'unknown')}"
                )
                cosmos_data["phase"] = f"checkpoint_{report_data['checkpoint/event']}"
                cosmos_data["checkpoint_path"] = report_data.get("checkpoint/path")
            elif "train/avg_loss" in report_data:
                message = f"Training complete - token-weighted loss: {report_data['train/avg_loss']:.6f}"
                cosmos_data["phase"] = "training_complete"
            elif "train/loss_avg" in report_data:
                message = f"Training {log_key} {current_value}/{max_value} - Loss: {report_data['train/loss_avg']:.6f}"
            elif "val/loss" in report_data or "val/avg_loss" in report_data:
                val_loss = report_data.get("val/loss", report_data.get("val/avg_loss"))
                message = f"Validation {log_key} {current_value}/{max_value} - Loss: {val_loss:.6f}"
                if "val/loss_numerator" in report_data:
                    cosmos_data["phase"] = "validation_complete"
            else:
                message = f"{self.experiment_name} in progress"

            # Write to status file
            # Check if using fallback logger (supports kpi as argument) or Framework logger
            # Framework StatusLogger: set kpi as attribute, then call write()
            try:
                from cosmos_framework.utils.workflow_status import Status, Verbosity

                status_logger.kpi = report_data
                status_logger.write(
                    data=cosmos_data,
                    status_level=Status.RUNNING,
                    verbosity_level=Verbosity.INFO,
                    message=message,
                )
            except Exception as write_err:
                logger.warning(f"Framework write failed: {write_err}")

            logger.debug(f"Cosmos status logged for {self.experiment_name} {log_key} {current_value}")

        except Exception as e:
            logger.warning(f"Cosmos status logging failed: {e}")

    def _write_status(self, phase: str, data: Dict[str, Any], message: str = "") -> None:
        """Write status update to Cosmos status file."""
        # Only write from master rank (rank 0)
        node_rank = int(os.environ.get("NODE_RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        if node_rank != 0 or local_rank != 0:
            return

        status_logger = self._get_status_logger()
        if status_logger is None:
            return

        try:
            # Convert tensors to scalars
            data = self._convert_tensors_to_scalars(data)

            # Only put component and phase at outer level, metrics go in kpi only
            cosmos_data = {
                "component": self.experiment_name,
                "phase": phase,
            }

            if not message:
                message = f"{self.experiment_name} - {phase}"

            try:
                from cosmos_framework.utils.workflow_status import Status, Verbosity

                status_logger.kpi = data
                status_logger.write(
                    data=cosmos_data,
                    status_level=Status.RUNNING,
                    verbosity_level=Verbosity.INFO,
                    message=message,
                )
            except Exception as write_err:
                logger.debug(f"Cosmos hook write failed: {write_err}")
        except Exception as e:
            logger.debug(f"Cosmos hook status failed: {e}")

    def get_hooks(self) -> Dict[str, Callable]:
        """Get all hooks for Cosmos status updates during training/validation.

        Returns hooks that write to the Cosmos status file for per-batch progress.
        """

        def pre_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status(
                "validation_starting",
                {
                    "validation_dataset_size": len(worker.val_data_loader.dataset),
                    **report_data,
                },
                f"Starting validation at epoch {report_data.get('current_epoch', 0) + 1}",
            )

        def pre_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            batch_idx = report_data.get("batch_index", 0)
            total_batches = len(worker.val_data_loader)
            progress = (batch_idx / total_batches) * 100 if total_batches > 0 else 0
            self._write_status(
                "validation_batch_start",
                {
                    "batch_index": batch_idx,
                    "total_batches": total_batches,
                    "progress_percent": progress,
                },
                f"Validation batch {batch_idx + 1}/{total_batches}",
            )

        def post_per_step_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            batch_idx = report_data.get("batch_index", 0)
            total_batches = len(worker.val_data_loader)
            progress = ((batch_idx + 1) / total_batches) * 100 if total_batches > 0 else 0
            self._write_status(
                "validation_batch_complete",
                {
                    "batch_index": batch_idx,
                    "total_batches": total_batches,
                    "progress_percent": progress,
                    "batch_loss": report_data.get("val_score"),
                },
                f"Validation batch {batch_idx + 1}/{total_batches} complete",
            )

        def post_validation_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status(
                "validation_complete",
                report_data,
                f"Validation complete. Avg loss: {report_data.get('val_avg_loss', 'N/A')}",
            )

        def pre_training_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status("training_starting", report_data)

        def post_training_hook(worker, report_data: Dict[str, Any]) -> None:
            self._write_status(
                "training_complete",
                report_data,
                f"Training complete. Avg loss: {report_data.get('train_avg_loss', 'N/A')}",
            )

        return {
            "pre_validation_hook": pre_validation_hook,
            "pre_per_step_validation_hook": pre_per_step_validation_hook,
            "post_per_step_validation_hook": post_per_step_validation_hook,
            "post_validation_hook": post_validation_hook,
            "pre_training_hook": pre_training_hook,
            "post_training_hook": post_training_hook,
        }
