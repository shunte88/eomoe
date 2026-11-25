# Multi-stage build for eomoe. CUDA 13.3 matches the host toolkit.
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH \
    CUDA_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY build.rs ./
COPY kernels/ kernels/
COPY src/ src/
COPY config.example.yaml ./
RUN cargo build --release

# Runtime image ships libcudart + cuBLAS; libcuda comes from the host driver via nvidia-container-toolkit.
FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/eomoe .
RUN mkdir -p /app/config /app/models

ENV EOMOE_CONFIG_PATH=/app/config/config.yaml
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["./eomoe", "serve", "--host", "0.0.0.0"]
