# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Names of the opt-in bitsandbytes eight-bit AdamW optimizers (non-paged, paged).

The single definition, torch-free and import-light, so configuration code can name the
optimizers without importing torch, bitsandbytes or ``cosmos_framework.utils.quantized_optimizer``.
"""

QUANTIZED_OPTIMIZER_NAMES: tuple[str, ...] = ("adamw_bnb_8bit", "paged_adamw_8bit")
