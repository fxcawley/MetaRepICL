# MetaRep — Reproducible CUDA environment
# Build: docker build -t metarep .
# Run:   docker run --gpus all -it metarep make smoke

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Use python3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /workspace

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Verify installation
RUN python -c "import torch; import numpy; import hydra; print('Install OK')"

# Default: run smoke test
CMD ["make", "smoke"]
