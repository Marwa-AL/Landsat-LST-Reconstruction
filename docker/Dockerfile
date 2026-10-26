# Reproduces the training environment of the published model:
# PyTorch 2.1.0 / CUDA 12.1 / cuDNN 8.9 / Python 3.10 / Ubuntu 20.04
#
# Build:  docker build -t lst-lama -f docker/Dockerfile .
# Run:    docker run --gpus all --shm-size=16g -it -v "$PWD":/workspace -w /workspace lst-lama
# enroot: see docs/SETUP.md (no Docker needed)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV MPLCONFIGDIR=/tmp/matplotlib \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
# torch / torchvision are pinned to the versions already in the image, so pip keeps them
RUN pip install -r /tmp/requirements.txt

WORKDIR /workspace
