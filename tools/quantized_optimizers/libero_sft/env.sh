# Shared cluster paths for the LIBERO-10 optimizer qualification (source with `.`).
# R is the authorized per-user project root; nothing here contains credentials.
export R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
export W=$R/cosmos-optimizer-libero-sft-20261116
export H=$R/hf-assets-20261116
export CONTAINER_IMAGE=$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh
export PYDEPS=$W/pydeps                       # bitsandbytes 0.49.2 (pinned project copy)
export LIBERO_ROOT=$H/LIBERO_LeRobot_v3/libero_10
export BASE_CHECKPOINT_PATH=$H/Cosmos3-Nano-dcp
export WAN_VAE_PATH=$H/wan22_vae/Wan2.2_VAE.pth
export RESOURCE_ROOT=$R
export SIM_PYTHON=$R/cosmos-optimizer-native-20261113/sim-runtime-v1/environment/bin/python
export SIM_OVERLAY=$W/sim-overlay              # requests/loguru/pynvml for the eval client only
