FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        gnupg \
        less \
        nano \
        sudo \
        unzip \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu

USER ubuntu
RUN git lfs install --skip-repo
WORKDIR /workspace
