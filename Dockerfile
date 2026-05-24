# Multi-stage build: CUDA-devel to compile whisper-rs with CUDA, CUDA-runtime for final image.

ARG CUDA_VER=12.4.1

############ chef base ############
FROM nvidia/cuda:${CUDA_VER}-devel-ubuntu22.04 AS chef
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config cmake clang libclang-dev git \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH=/root/.cargo/bin:$PATH
RUN cargo install cargo-chef --locked
WORKDIR /app

############ planner ############
FROM chef AS planner
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

############ builder ############
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# Build deps first (cached layer)
RUN cargo chef cook --release --recipe-path recipe.json
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release --bin rust-whisper-mcp

############ runtime ############
FROM nvidia/cuda:${CUDA_VER}-runtime-ubuntu22.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: apt packages (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg python3 python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: deno — required by yt-dlp for YouTube JS extraction (independently cacheable)
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh

# Layer 3: pip packages (independently cacheable)
RUN pip3 install --no-cache-dir yt-dlp

WORKDIR /app
COPY --from=builder /app/target/release/rust-whisper-mcp /usr/local/bin/rust-whisper-mcp
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV RUST_LOG=info \
    WHISPER_MODEL_PATH=/models/ggml-large-v3.bin \
    WHISPER_MODEL_URL=https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin \
    DATA_DIR=/data \
    BIND_ADDR=0.0.0.0:8080

EXPOSE 8080
VOLUME ["/models", "/data"]
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
