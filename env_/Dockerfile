FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    python3 \
    python3-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 更新 pip
RUN pip install pip --upgrade && \
    pip install wheel

# 安装 PyTorch
RUN pip install tqdm \
    && pip3 install --progress-bar=on --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

COPY . .

RUN pip install -r requirements.txt


# RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install -r requirements.txt -i https://download.pytorch.org/whl/nightly/cu118


# FROM nvidia/cuda:12.0-base-ubuntu20.04

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         python3-pip \
#         python3-dev \
#         python3-opencv \
#         libglib2.0-0
# Install any python packages you need
# COPY . .

# RUN python3 -m pip install -r requirements.txt
# RUN pip install -r requirements.txt

# Upgrade pip
# RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
# RUN pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118