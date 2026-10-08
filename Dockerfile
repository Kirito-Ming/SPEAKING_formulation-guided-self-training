FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip git ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/fgst
COPY pyproject.toml ./
COPY fgst ./fgst
RUN python3 -m pip install --upgrade pip && python3 -m pip install .
COPY . .
ENTRYPOINT ["/bin/bash"]
