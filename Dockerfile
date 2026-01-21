# Short Video Agent - GPU-enabled Docker image
# Supports LTX-2 local video generation and Replicate API fallback
#
# Build: docker build -t short-video-agent .
# Run:   docker run --gpus all -v $(pwd)/outputs:/app/outputs short-video-agent

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .
COPY requirements-gpu.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-gpu.txt

# Copy source code
COPY src/ ./src/
COPY schemes/ ./schemes/
COPY speakers/ ./speakers/
COPY products/ ./products/

# Copy entry points
COPY run.py .

# Create output directory
RUN mkdir -p outputs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
ENTRYPOINT ["python", "run.py"]
CMD ["--help"]
