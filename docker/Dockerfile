FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04

MAINTAINER Yutaro Shigeto <shigeto@stair.center>

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    parallel \ 
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install --upgrade setuptools

WORKDIR /tmp
# pytorch
RUN python3.8 -m pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
# requireqments
COPY requirements.txt /tmp/requirements.txt
RUN python3.8 -m pip install -r requirements.txt
# dali
RUN python3.8 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110==1.16.0
RUN python3.8 -m pip install matplotlib==3.7.1 seaborn==0.11.2 pandas==2.0.2 numba==0.56.0 umap-learn==0.5.3

ENV PYTHONIOENCODING=utf-8
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# wandb
RUN git config --global --add safe.directory /workspace
ENTRYPOINT wandb login $WANDB_API_KEY && /bin/bash
