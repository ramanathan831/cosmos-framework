# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1

# BASE_IMAGE must be built from the pinned official Cosmos Predict 2.5 source,
# retaining its checkout and Python environment. Supply their paths at startup.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY . /opt/cosmos-framework
RUN python -m pip install --no-deps /opt/cosmos-framework && \
    python -m pip install 'fastapi>=0.115' 'uvicorn>=0.34'
ENTRYPOINT ["cosmos-predict-serve"]
