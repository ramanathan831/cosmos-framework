# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Build with this Framework checkout as context and a clean native Cosmos-RL
# checkout as the named cosmos-rl context. BASE_IMAGE must supply CUDA/PyTorch.
# No bundled toolkit, private action image, or dataset package is used.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG SOURCE_COMMIT
ARG SOURCE_TREE
ARG SOURCE_DIRTY=1
ARG BUILD_TIMESTAMP
ARG COSMOS_RL_COMMIT
ARG COSMOS_RL_TREE
ENV SOURCE_COMMIT=${SOURCE_COMMIT} SOURCE_TREE=${SOURCE_TREE} \
    SOURCE_DIRTY=${SOURCE_DIRTY} BUILD_TIMESTAMP=${BUILD_TIMESTAMP} \
    COSMOS_RL_COMMIT=${COSMOS_RL_COMMIT} COSMOS_RL_TREE=${COSMOS_RL_TREE} \
    PROVENANCE_BASE_IMAGE=${BASE_IMAGE}
COPY --from=cosmos-rl / /opt/cosmos-rl
COPY . /workspace
RUN python3 -m venv --system-site-packages /opt/venv/cosmos_rl && \
    /opt/venv/cosmos_rl/bin/python -m pip install '/workspace[workflows]' /opt/cosmos-rl \
        'qwen-vl-utils==0.0.14' 'PyNvVideoCodec>=2.0' 'cuda-python>=12' && \
    /opt/venv/cosmos_rl/bin/python -m cosmos_framework.integrations.cosmos_rl.runtime_dependency_contract \
        --repair-qwen-pynv-worker && \
    /opt/venv/cosmos_rl/bin/python /workspace/docker/write_image_provenance.py && \
    /opt/venv/cosmos_rl/bin/python -m cosmos_framework.scripts.prepare_vlm_checkpoint \
        --cosmos-runtime-preflight > /opt/cosmos/framework-converter-runtime.json
ENV PATH=/opt/venv/cosmos_rl/bin:${PATH}
WORKDIR /workspace
CMD ["/bin/bash"]
